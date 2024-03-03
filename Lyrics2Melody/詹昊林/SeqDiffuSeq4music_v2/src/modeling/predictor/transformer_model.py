from transformers import AutoConfig
from modeling_bart import BartModel, BartAttention
import torch
import torch as th
import torch.nn as nn
from src.modeling.diffusion.nn import (
    SiLU,
    linear,
    timestep_embedding,
)
import math

from einops import rearrange, repeat
from typing import List, Optional, Tuple, Union
from transformers.models.bart.configuration_bart import BartConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=None, drop=0):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden_features)
        self.act_layer = act_layer if act_layer else nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(drop)
        self.linear2 = torch.nn.Linear(hidden_features, in_features)
    def forward(self, x):
            x = self.linear1(x)
            x = self.act_layer(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x

# class FullAttention(nn.Module):
#     '''
#     The Attention operation
#     '''
#     def __init__(self, scale=None, attention_dropout=0.1):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.dropout = nn.Dropout(attention_dropout)
        
#     def forward(self, queries, keys, values):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1./math.sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)
        
#         return V.contiguous()

# class AttentionLayer(nn.Module):
#     '''
#     The Multi-head Self-Attention (MSA) Layer
#     '''
#     def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model//n_heads)
#         d_values = d_values or (d_model//n_heads)

#         self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.mix = mix

#     def forward(self, queries, keys, values):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out = self.inner_attention(
#             queries,
#             keys,
#             values,
#         )
#         if self.mix:
#             out = out.transpose(2,1).contiguous()
#         out = out.view(B, L, -1)

#         return self.out_projection(out)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length = 0):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_no_causal_mask(
            input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
        # ).to(self.device)
        ).to(inputs_embeds.device)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

def _make_no_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 0.)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        approx_gelu = nn.GELU()
        self.mlp = MLP(in_features=self.embed_dim, hidden_features=config.decoder_ffn_dim, act_layer=approx_gelu, drop=self.dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition_states).chunk(6, dim=-1)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )


        hidden_states = residual + gate_msa * hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + gate_mlp * self.mlp(modulate(hidden_states, shift_mlp, scale_mlp))

        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# class DiTBlock(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=2.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.attn = AttentionLayer(d_model=hidden_size, n_heads=num_heads)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = nn.GELU()
#         # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
#         h=modulate(self.norm1(x), shift_msa, scale_msa)
#         # x = x + gate_msa.unsqueeze(1) * self.attn(h,h,h)
#         # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         x = x + gate_msa * self.attn(h,h,h)
#         x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x

def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """
#     def __init__(self, hidden_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.fc = nn.Sequential(
#             nn.Tanh(),
#             nn.Linear(hidden_size, out_channels, bias=True)
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
#         # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         h = modulate(self.norm_final(x), shift, scale)
#         # # h = modulate(x, shift, scale)
#         x = self.fc(x + h)
#         # x = self.fc(h)
#         return x


class TransformerNetModel_encoder_decoder(nn.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        init_pretrained,
        freeze_embeddings,
        use_pretrained_embeddings,
        dropout=0,
        use_checkpoint=False,
        num_heads=1,
        config=None,
        config_name="bert-base-uncased",
        vocab_size=None,
        logits_mode=1,
        encoder_layers = 6,
        decoder_layers = 6,
        load_ckpt=None,
        DiT_depth = 2,
        num_classes = 1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.dropout = dropout
            # config.hidden_size = 512

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained = init_pretrained
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.config_name = config_name
        self.load_ckpt = load_ckpt

        if not self.init_pretrained:
            self.config.encoder_layers = encoder_layers
            self.config.decoder_layers = decoder_layers
            self.config.vocab_size = vocab_size
            self.config.encoder_attention_heads = num_heads
            self.config.decoder_attention_heads = num_heads
            self.config.d_model = in_channels
            self.config.encoder_ffn_dim = model_channels
            self.config.decoder_ffn_dim = model_channels
            self.embedding_dim = int(self.config.d_model // 4)
            self.embed_scale = math.sqrt(self.embedding_dim) if self.config.scale_embedding else 1.0

        time_embed_dim = in_channels
        self.time_embed = nn.Sequential(
            linear(in_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.d_model),
        )


        self.build_xstart_predictor()
        self.build_input_output_projections()
        self.build_embeddings()

        self.LayerNorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if self.load_ckpt is not None:
            self.load_weight(self.load_ckpt)


        if num_classes>1: self.y_embedder = LabelEmbedder(num_classes, config.d_model, config.dropout)
        self.DiT_depth = DiT_depth
        self.DiT_blocks = nn.ModuleList([
            DiTDecoderLayer(self.config) for _ in range(self.DiT_depth)
        ])


    def get_embeds(self, input_ids):
        return self.input_transformers.decoder.embed_tokens(input_ids) * self.embed_scale

    def load_weight(self, path):

        self.load_state_dict(torch.load(path))
        print(f'weigth initialize from {path}')

    def build_xstart_predictor(self):
        if self.init_pretrained:

            temp_bart = BartModel.from_pretrained(self.config_name, config=self.config)
            self.input_transformers = temp_bart
        else:
            self.input_transformers = BartModel(self.config, self.embedding_dim)

    def build_input_output_projections(self):
        if self.in_channels != self.embedding_dim:
             # need to adapt the model to the embedding size
            self.input_up_proj_dec = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )
                
            self.input_up_proj_enc = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.config.d_model),
                    nn.Tanh(),
                    nn.Linear(self.config.d_model, self.config.d_model),
                )

            self.output_down_proj = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.embedding_dim),
            )
            # self.output_down_proj = FinalLayer(self.config.d_model, self.embedding_dim)
        else:
            self.input_up_proj = nn.Identity()
            self.output_down_proj = nn.Identity()


    def build_embeddings(self):

        self.lm_head = nn.Linear(self.embedding_dim, self.input_transformers.shared.weight.shape[0])

        with th.no_grad():
            self.lm_head.weight = self.input_transformers.shared.weight

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward_encoder(self, 
                input_ids = None,
                timesteps = None,
                attention_mask = None,
                decoder_inputs_embeds = None,
                decoder_attention_mask = None,
                self_conditions = None,
                y=None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        # decoder_inputs_embeds = self.input_transformers.decoder.embed_tokens(decoder_input_ids) * self.embed_scale
        if self_conditions is not None:
            
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim = -1)

        decoder_inputs_embeds = (
            self.input_up_proj_dec(decoder_inputs_embeds)
            + emb
        )
        
        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))
        
        encoder_hidden_states = self.input_transformers(
            input_ids = None,
            attention_mask=attention_mask,
            inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale),
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs, 
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        ).encoder_last_hidden_state
        
        return encoder_hidden_states

    def forward(self, 
                input_ids = None,
                timesteps = None,
                attention_mask = None,
                decoder_inputs_embeds = None,
                decoder_attention_mask = None,
                self_conditions = None,
                encoder_outputs=None,
                y=None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert encoder_outputs is None or input_ids is None
        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        if self_conditions is not None:
            
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim = -1)

        decoder_inputs_embeds = (
            self.input_up_proj_dec(decoder_inputs_embeds)
            + emb
        )

        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))
        
        # input_trans_hidden_states = self.input_transformers(
        #     input_ids = None,
        #     attention_mask=attention_mask,
        #     inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale) if input_ids is not None else None,
        #     decoder_input_ids=None,
        #     decoder_inputs_embeds=emb_inputs, 
        #     decoder_attention_mask=decoder_attention_mask,
        #     encoder_outputs=encoder_outputs
        # ).last_hidden_state
        

        input_trans_dict = self.input_transformers(
            input_ids = None,
            attention_mask=attention_mask,
            inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale) if input_ids is not None else None,
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs, 
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs
        )
        input_trans_hidden_states=input_trans_dict.last_hidden_state
        encoder_outputs=(input_trans_dict.encoder_last_hidden_state,
            input_trans_dict.encoder_hidden_states,
            input_trans_dict.encoder_attentions)

        return_dict = self.config.use_return_dict
        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_hidden_states = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            encoder_hidden_states=encoder_hidden_states[0]
        else: encoder_hidden_states=encoder_outputs[0]
        
        input_shape = input_trans_hidden_states.size()[:-1]
        DiTdecoder_attention_mask = _prepare_decoder_attention_mask(
            decoder_attention_mask, input_shape, input_trans_hidden_states
        )

        DiT_gradient_checkpointing=False
        DiT_use_cache=False
        DiT_output_attentions=False
        DiT_next_decoder_cache=()
        DiT_all_self_attns=()
        DiT_cross_attentions=()

        # expand encoder attention mask
        if encoder_hidden_states is not None and attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            DiTencoder_attention_mask = _expand_mask(attention_mask, input_trans_hidden_states.dtype, tgt_len=input_shape[-1])

        if y is not None: y = self.y_embedder(y, self.training)    # (N, D)
        else: y=0
        c = emb + y                                # (N, D)
        for DiT_block in self.DiT_blocks:
            # input_trans_hidden_states = block(input_trans_hidden_states, c)                      # (N, T, D)

            # # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # if output_hidden_states:
            #     all_hidden_states += (hidden_states,)
            # dropout_probability = random.uniform(0, 1)
            # if self.training and (dropout_probability < self.layerdrop):
            #     continue

            # past_key_value = past_key_values[idx] if past_key_values is not None else None

            if DiT_gradient_checkpointing and self.training:

                if DiT_use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    DiT_use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, DiT_output_attentions, DiT_use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = DiT_block(
                    hidden_states=input_trans_hidden_states,
                    condition_states=c,
                    attention_mask=DiTdecoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=DiTencoder_attention_mask,
                    output_attentions=self.config.output_attentions,
                    use_cache=DiT_use_cache,
                )
                input_trans_hidden_states = layer_outputs[0]

            if DiT_use_cache:
                DiT_next_decoder_cache += (layer_outputs[3 if DiT_output_attentions else 1],)

            if DiT_output_attentions:
                DiT_all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    DiT_cross_attentions += (layer_outputs[2],)


        h = self.output_down_proj(input_trans_hidden_states)
        # h = self.output_down_proj(input_trans_hidden_states, c)

        return h
