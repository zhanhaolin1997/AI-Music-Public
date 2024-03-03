import enum
import math
from typing import Optional

import torch
from torch import nn

from fairseq.models.nat import NATransformerDecoder
from fairseq.models.transformer import TransformerEncoder, TransformerConfig
from fairseq.modules import PositionalEmbedding, LayerDropModuleList


from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from typing import Dict, List, Optional
from torch import Tensor
from fairseq import utils
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


from improved_diffusion.nn import timestep_embedding

from .utils import build_ffn


class adaLN_module(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    # def __init__(self, hidden_size, num_heads, mlp_ratio=2.0, **block_kwargs):
    def __init__(self, hidden_size, condition_size, attn_func):
        super().__init__()
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.attn = AttentionLayer(d_model=hidden_size, n_heads=num_heads)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp = build_ffn(hidden_size, (hidden_size+hidden_size), hidden_size, 'gelu')
        # self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )
        self.adaLN_modulation = build_ffn(condition_size, (6*hidden_size + condition_size), 6*hidden_size, 'silu')
        self.attn_func = attn_func

    def modulate(self, x, shift, scale):
        return x * (1 + scale.squeeze(1)) + shift.squeeze(1)
        # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c, attn_type, **attn_para):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        h = self.modulate(x, shift_msa, scale_msa)
        if attn_type=='self_attn': h, attn = self.attn_func(query=h,
                                                        key=attn_para['key'],
                                                        value=attn_para['value'],
                                                        key_padding_mask=attn_para['key_padding_mask'],
                                                        incremental_state=attn_para['incremental_state'],
                                                        need_weights=attn_para['need_weights'],
                                                        attn_mask=attn_para['attn_mask'],
                                                        )
        elif attn_type=='encoder_attn': h, attn = self.attn_func(query=h,
                                                            key=attn_para['key'],
                                                            value=attn_para['value'],
                                                            key_padding_mask=attn_para['key_padding_mask'],
                                                            incremental_state=attn_para['incremental_state'],
                                                            static_kv=attn_para['static_kv'],
                                                            need_weights=attn_para['need_weights'],
                                                            need_head_weights=attn_para['need_head_weights'],
                                                            )
        x = x + gate_msa.squeeze(1) * h
        x = x + gate_mlp.squeeze(1) * self.mlp(self.modulate(x, shift_mlp, scale_mlp))
        # h=modulate(self.norm1(x), shift_msa, scale_msa)
        # x = x + gate_msa.unsqueeze(1) * self.attn(h,h,h)
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # return x
        return x, attn

class DiffusionTransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = (
            LayerNorm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
            if scale_heads
            else None
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.ffn_layernorm = (
            LayerNorm(cfg.decoder.ffn_embed_dim)
            if utils.safe_getattr(cfg, "scale_fc", False)
            else None
        )
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if utils.safe_getattr(cfg, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False


        self.self_attn_adaLN=self.build_adaLN_module(embed_dim=self.embed_dim, cond_dim=cfg.model_dim, attn_func=self.self_attn)
        self.encoder_attn_adaLN=self.build_adaLN_module(embed_dim=self.embed_dim, cond_dim=cfg.model_dim, attn_func=self.encoder_attn)
    
    def build_adaLN_module(
            self, embed_dim, cond_dim, attn_func,
        ):
        return adaLN_module(embed_dim, cond_dim, attn_func)


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        c,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        # x, attn = self.self_attn(
        #     query=x,
        #     key=y,
        #     value=y,
        #     key_padding_mask=self_attn_padding_mask,
        #     incremental_state=incremental_state,
        #     need_weights=False,
        #     attn_mask=self_attn_mask,
        # )


        x, attn = self.self_attn_adaLN(
                x,
                c,
                attn_type='self_attn',
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )


        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # x, attn = self.encoder_attn(
            #     query=x,
            #     key=encoder_out,
            #     value=encoder_out,
            #     key_padding_mask=encoder_padding_mask,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )


            x, attn = self.encoder_attn_adaLN(
                x,
                c,
                attn_type='encoder_attn',
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )


            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

class DifformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None):
        super().__init__(args, dictionary, embed_tokens)

        self.project_in_dim = project_in_dim

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        
        x = embed = self.embed_scale * token_embedding
        x = self.project_in_dim(x)

        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


class EmbedNormPosition(enum.Enum):
    NO_EMBED_NORM = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class SelfCondPosition(enum.Enum):
    NO_SELF_COND = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class DifformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None, project_out_dim=None, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens)

        latent_dim = args.latent_dim
        model_dim = args.model_dim
        
        self.project_in_dim = project_in_dim
        self.project_out_dim = project_out_dim

        # embedding normalization
        if not args.embed_norm:
            args.embed_norm_position = EmbedNormPosition.NO_EMBED_NORM
        elif args.embed_norm_before_proj:
            args.embed_norm_position = EmbedNormPosition.BEFORE_PROJ
        else:
            args.embed_norm_position = EmbedNormPosition.AFTER_PROJ
        
        if args.embed_norm:
            self.embed_norm = nn.LayerNorm(
                latent_dim if args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ
                else model_dim,
                elementwise_affine=args.embed_norm_affine
            )

        # self-conditioning
        if not args.self_cond:
            args.self_cond_position = SelfCondPosition.NO_SELF_COND
        elif args.self_cond_before_proj:
            args.self_cond_position = SelfCondPosition.BEFORE_PROJ
        else:
            args.self_cond_position = SelfCondPosition.AFTER_PROJ
        
        if args.self_cond:
            self_cond_dim = (
                latent_dim if args.self_cond_position == SelfCondPosition.BEFORE_PROJ
                else model_dim
            )

            self.self_cond_proj = build_ffn(
                self_cond_dim * 2, self_cond_dim, self_cond_dim,
                args.activation_fn, args.dropout
            )

        self.embed_time = build_ffn(model_dim, model_dim * 4, model_dim, args.activation_fn)

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.latent_dim)
        

        # cfg = TransformerConfig.from_namespace(args)
        # self.decoder_layerdrop = cfg.decoder.layerdrop
        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(self.cfg, no_encoder_attn)
                for _ in range(self.cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)


    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = DiffusionTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


    def forward_embedding(self, tokens):
        embed = self.embed_scale * self.embed_tokens(tokens)

        if self.args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ:
            embed = self.embed_norm(embed)
        
        return embed

    def forward(self, z_t, t, mask, encoder_out, prev_z_0_hat=None, **kwargs):
        hidden, time_emb = self.forward_hidden(z_t, t, mask, prev_z_0_hat)

        # B x T x C -> T x B x C
        hidden = hidden.transpose(0, 1)
        attn = None
        inner_states = [hidden]

        # decoder layers
        for i, layer in enumerate(self.layers):
            hidden, attn, _ = layer(
                hidden,
                time_emb,
                # encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_out'][-1] if encoder_out is not None else None, #TO DO 不知道为什么encoder_out['encoder_out']被list包裹着, 所以取个[-1]先用着
                # encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'][-1] if encoder_out is not None else None, #TO DO 不知道为什么encoder_out['encoder_padding_mask']被list包裹着, 所以取个[-1]先用着
                self_attn_mask=None,
                self_attn_padding_mask=~mask,
            )
            inner_states.append(hidden)

        if self.layer_norm:
            hidden = self.layer_norm(hidden)

        # T x B x C -> B x T x C
        hidden = hidden.transpose(0, 1)

        hidden = self.project_out_dim(hidden)
        return hidden, {"attn": attn, "inner_states": inner_states}

    def forward_hidden(self, z_t, t, mask, prev_z_0_hat=None):
        # self-conditioning
        if self.args.self_cond_position == SelfCondPosition.BEFORE_PROJ:
            cat_embed = torch.cat((z_t, prev_z_0_hat), -1)
            hidden = self.project_in_dim(self.self_cond_proj(cat_embed))
        
        elif self.args.self_cond_position == SelfCondPosition.AFTER_PROJ:
            z_hidden = self.project_in_dim(z_t)
            prev_hidden = self.project_in_dim(prev_z_0_hat)
            cat_hidden = torch.cat((z_hidden, prev_hidden), -1)
            hidden = self.self_cond_proj(cat_hidden)
        
        else:
            hidden = self.project_in_dim(z_t)

        # time embedding
        time_emb = self.embed_time(timestep_embedding(t, self.args.model_dim).type_as(z_t))[:, None]
        hidden = hidden + time_emb

        # position embedding
        positions = self.embed_positions(mask.long() + self.padding_idx)
        hidden = hidden + positions
        
        # embedding normalization
        if self.args.embed_norm_position == EmbedNormPosition.AFTER_PROJ:
            hidden = self.embed_norm(hidden)
        
        hidden = self.dropout_module(hidden)
        return hidden, time_emb
