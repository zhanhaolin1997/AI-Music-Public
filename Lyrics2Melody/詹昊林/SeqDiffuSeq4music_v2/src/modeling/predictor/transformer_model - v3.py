from transformers import AutoConfig
from modeling_bart import BartModel
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

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)

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


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=2.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = AttentionLayer(d_model=hidden_size, n_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = nn.GELU()
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        h=modulate(self.norm1(x), shift_msa, scale_msa)
        # x = x + gate_msa.unsqueeze(1) * self.attn(h,h,h)
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_msa * self.attn(h,h,h)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        h = modulate(self.norm_final(x), shift, scale)
        # # h = modulate(x, shift, scale)
        x = self.fc(x + h)
        # x = self.fc(h)
        return x


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
        self.DiT_num_heads = num_heads
        self.DiT_blocks = nn.ModuleList([
            DiTBlock(config.d_model, self.DiT_num_heads) for _ in range(self.DiT_depth)
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

            # self.output_down_proj = nn.Sequential(
            #     nn.Linear(self.config.d_model, self.config.d_model),
            #     nn.Tanh(),
            #     nn.Linear(self.config.d_model, self.embedding_dim),
            # )
            self.output_down_proj = FinalLayer(self.config.d_model, self.embedding_dim)
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
        
        input_trans_hidden_states = self.input_transformers(
            input_ids = None,
            attention_mask=attention_mask,
            inputs_embeds = self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale) if input_ids is not None else None,
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs, 
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs
        ).last_hidden_state
        

        if y is not None: y = self.y_embedder(y, self.training)    # (N, D)
        else: y=0
        c = emb + y                                # (N, D)
        for block in self.DiT_blocks:
            input_trans_hidden_states = block(input_trans_hidden_states, c)                      # (N, T, D)


        # h = self.output_down_proj(input_trans_hidden_states)
        h = self.output_down_proj(input_trans_hidden_states, c)

        return h
