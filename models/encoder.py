import torch
from torch import nn

from models.embedder import PositionEncoding
from models.attentions import MultiHeadRSA, Attention
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
        self.pe = PositionEncoding(n_filters=768)
        self.layers = nn.ModuleList([TrmEncLayer(self.cfg) for _ in range(num_layers)])

    def forward(self, x):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)

        return x


class TrmEncLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = MultiHeadRSA(cfg)
        self.hidden_size = cfg.hidden_size

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
        tmp_x = x.clone().cuda()      # (16, 7, 768)
        target = tmp_x.clone().cuda() # (16, 7, 768)
        target = self.attention(target, tmp_x)
        x = self.z * tmp_x + (1 - self.z) * target
        x = self.LayerNorm(x)
        tmp_x = x.clone().cuda()
        x = self.ffn(x)
        x = self.rand * tmp_x + (1 - self.rand) * x
        x = self.LayerNorm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList(
            [LayerWoMemory(cfg) for _ in range(cfg.num_hidden_layers)]
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
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, clip_feats)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class LayerWoMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.att_num = 2
        self.attention = Attention(cfg)
        self.hidden_intermediate = Intermediate(cfg)
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.rand_z = torch.randn(1, requires_grad=True).cuda()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, clip_feats=None):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.cfg.max_v_len, self.cfg.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(
            attention_mask, max_v_len, max_t_len
        )  # (N, L, L)
        tmp_x = hidden_states.clone().cuda()
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.attention(hidden_states, shifted_self_mask, clip_feats)
        hidden_states = self.rand * tmp_x + (1 - self.rand) * hidden_states
        tmp_x = hidden_states.clone().cuda()
        hidden_states = self.LayerNorm(hidden_states)
        intermediate_output = self.hidden_intermediate(hidden_states)
        layer_output = self.rand_z * tmp_x + (1 - self.rand_z) * intermediate_output
        return layer_output
