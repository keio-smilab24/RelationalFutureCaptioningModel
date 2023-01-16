import torch
from torch import nn

from models.embedder import PositionEncoding
from models.attentions import MultiHeadRSA, MultiHeadAttention
from models.misc import FeedforwardNeuralNetModel, make_pad_shifted_mask, Intermediate


class RSAEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.TSEncoder = TimeSeriesEncoder(self.cfg)
        self.expand = nn.Linear(self.cfg.hidden_size, self.hidden_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.z = torch.randn(1, requires_grad=True).cuda()

    def forward(self, x):
        ts_feats = x.clone().cuda()
        ts_feats = self.TSEncoder(ts_feats)
        ts_feats = self.z * x + (1 - self.z) * ts_feats
        ts_feats = self.expand(ts_feats)
        tmp_feats = ts_feats[:, 1, :].reshape((-1, 1, self.hidden_size))
        tmp_feats = self.layernorm(tmp_feats)
        return ts_feats, tmp_feats


class TimeSeriesEncoder(nn.Module):
    def __init__(self, cfg, num_layers=2):
        super().__init__()
        self.cfg = cfg
        # self.pe = PositionEncoding(n_filters=768)
        self.layers = nn.ModuleList([TrmEncLayer(self.cfg) for _ in range(num_layers)])
        # self.ff = TrmFeedForward(self.cfg)

    def forward(self, x):
        # x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        # x = self.ff(x, x)

        return x


class TrmEncLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        # self.attention = RelationalSelfAttention(cfg)
        # self.attention = Attention(cfg)
        self.mhrsa = MultiHeadRSA(cfg)
        # self.output = TrmFeedForward(cfg)  # 全結合層

        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.z = torch.randn(1, requires_grad=True).cuda()
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.ffn = FeedforwardNeuralNetModel(self.hidden_size, self.hidden_size * 2, self.hidden_size)

    def forward(self, x):
        """
        Args:
            x: (N, L, D)
        Returns:
        """
        # x = self.LayerNorm(x)
        identity_x = x.clone().cuda()      # (16, 7, 768)
        target = identity_x.clone().cuda() # (16, 7, 768)
        target = self.mhrsa(target, identity_x)
        x = self.z * identity_x + (1 - self.z) * target
        x = self.LayerNorm(x)
        identity_x = x.clone().cuda()
        # x = self.LayerNorm(x)
        x = self.ffn(x)
        x = self.rand * identity_x + (1 - self.rand) * x
        x = self.LayerNorm(x)
        # x = self.attention(x)
        # x = self.output(x, x)  # (N, L, D)
        return x

class CrossAttentionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList(
            [CAEncoderLayer(cfg) for _ in range(cfg.cross_attention_layers)]
        )

    def forward(
        self,
        hidden_states,
        source_kv,
        attention_mask=None,
        ):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step.
            Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        for _, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, source_kv, attention_mask)
        return hidden_states

class CAEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        
        # source - target
        self.crossmha = MultiHeadAttention(cfg)
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.linear = Intermediate(cfg)
        self.rand_z = torch.randn(1, requires_grad=True).cuda()

        # self attention
        self.selfmha = MultiHeadAttention(cfg)
        self.rand2 = torch.randn(1, requires_grad=True).cuda()
        self.linear2 = Intermediate(cfg)
        self.rand_z2 = torch.randn(1, requires_grad=True).cuda()
    
    def forward(self, x:torch.Tensor, source_kv:torch.Tensor, attention_mask=None):
        # souce-target
        identity_x = x.clone().cuda()
        x = self.crossmha(x=x, source_kv=source_kv)
        x = self.rand*identity_x + (1-self.rand)*x
        x = self.LayerNorm(x)
        identity_x = x.clone().cuda()
        x = self.linear(x)
        x = self.rand_z*identity_x + (1-self.rand_z)*x
        x = self.LayerNorm(x)

        # self attention
        identity_x = x.clone().cuda()
        x = self.selfmha(x)
        x = self.rand2*identity_x + (1-self.rand2)*x
        x = self.LayerNorm(x)
        identity_x = x.clone().cuda()
        x = self.linear2(x)
        x = self.rand_z2*identity_x + (1-self.rand_z2)*x
        x = self.LayerNorm(x)

        return x



class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers: bool=True,
        clip_feats=None):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step.
            Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, clip_feats)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.att_num = 2
        # self.attention = nn.ModuleList(
        #     [Attention(cfg) for _ in range(self.att_num)]
        # )
        # self.mmha = Attention(cfg)
        # self.mha = Attention(cfg)
        self.attention = MultiHeadAttention(cfg)
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.hidden_intermediate = Intermediate(cfg)
        self.rand_z = torch.randn(1, requires_grad=True).cuda()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        # self.output = Output(cfg)
        # self.ffn = FeedforwardNeuralNetModel(cfg.hidden_size, cfg.hidden_size * 2, cfg.hidden_size)

    def forward(self, hidden_states, attention_mask, clip_feats=None):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.cfg.max_v_len, self.cfg.max_t_len
        
        # self-attention, need to shift right # (N, L, L)
        # make mask
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)
        
        # attention layer
        tmp_x = hidden_states.clone().cuda()
        hidden_states = self.attention(hidden_states, shifted_self_mask, clip_feats)
        hidden_states = self.rand * tmp_x + (1 - self.rand) * hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        
        # ffn layer
        tmp_x = hidden_states.clone().cuda()
        intermediate_output = self.hidden_intermediate(hidden_states)
        hidden_states = self.rand_z * tmp_x + (1 - self.rand_z) * intermediate_output
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
