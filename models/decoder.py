import torch
from torch import nn
import torch.nn.functional as F

from models.attentions import MultiHeadAttention
from models.misc import FeedforwardNeuralNetModel, make_pad_shifted_mask, Intermediate
from models.encoder import CrossAttentionEncoder

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # attention
        self.attention = MultiHeadAttention(cfg)
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.rand2 = torch.randn(1, requires_grad=True).cuda()

        # ffn
        self.ffn = FeedforwardNeuralNetModel(cfg.hidden_size, cfg.hidden_size * 2, cfg.hidden_size)
        self.rand_z = torch.randn(1, requires_grad=True).cuda()

        # self attention
        self.selfmha = MultiHeadAttention(cfg)
        self.rand3 = torch.randn(1, requires_grad=True).cuda()
        self.linear2 = Intermediate(cfg)
        self.rand_z2 = torch.randn(1, requires_grad=True).cuda()


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_his: torch.Tensor,
        make_knn_dstore: bool=False
        ):
        """
        # attention layer (default)
        identity_x = x.clone().cuda()
        att = self.attention(x=x, source_kv=clip_his)
        x = self.rand*identity_x + (1 - self.rand)*att
        x = self.LayerNorm(x)
        """



        # attention layer (img : kv, text : q)
        identity_x = x.clone().cuda()
        x_1 = x.clone().cuda()
        att = self.attention(x=x_1, source_kv=clip_his)
        x_1 = self.rand*identity_x + (1 - self.rand)*att
        x_1 = self.LayerNorm(x_1)

        """
        # self attention
        identity_x = x_1.clone().cuda()
        x_1 = self.selfmha(x_1)
        x_1 = self.rand2*identity_x + (1-self.rand2)*x_1
        x_1 = self.LayerNorm(x_1)
        """

        # attention layer (img : q, text : kv)
        identity_clip_his = clip_his.clone().cuda()
        att = self.attention(x=clip_his, source_kv=x)
        clip_his = self.rand2*identity_clip_his + (1 - self.rand2)*att
        clip_his = self.LayerNorm(clip_his)

        """
        # self attention
        identity_x = x_2.clone().cuda()
        x_2 = self.selfmha(x_2)
        x_2 = self.rand2*identity_x + (1-self.rand2)*x_2
        x_2 = self.LayerNorm(x_2)
        '"""
        # attention layer (x_1 : kv, x_2 : q)
        identity_x = x_1.clone().cuda()
        att = self.attention(x=x_1, source_kv=clip_his)
        x = self.rand3*identity_x + (1 - self.rand3)*att
        x = self.LayerNorm(x)

        """
        # self attention
        identity_x = x.clone().cuda()
        x = self.selfmha(x)
        x = self.rand2*identity_x + (1-self.rand2)*x
        x = self.LayerNorm(x)
        """

        if make_knn_dstore: #最後の文字なら
            knn_feat = x.clone().detach().cpu()

        # ffn
        identity_x = x.clone().cuda()
        x = self.ffn(x)
        output = self.rand_z*identity_x + (1 - self.rand_z)*x
        output = self.LayerNorm(output)

        if make_knn_dstore: #最後の文字なら
            return output, knn_feat

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, num_hidden_layers=5):
        super().__init__()
        self.layer = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_his: torch.Tensor,
        make_knn_dstore: bool=False
        ):
        all_layer_outputs = []
        knn_feats = []

        for layer in self.layer:
            if make_knn_dstore:
                hidden_states, knn_feat = layer(hidden_states, attention_mask, clip_his, make_knn_dstore)
                knn_feats.append(knn_feat)
            else:
                hidden_states = layer(hidden_states, attention_mask, clip_his)
            all_layer_outputs.append(hidden_states)

        if make_knn_dstore:
            return all_layer_outputs, knn_feats
        else:
            return all_layer_outputs


class TrmFeedForward(nn.Module):
    """
    TransformerにおけるFF層
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense_f = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.dense_s = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense_f(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense_s(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PredictionHeadTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.gelu = nn.GELU()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PredictionHead(nn.Module):
    def __init__(self, cfg, bert_model_embedding_weights=None):
        super().__init__()
        self.transform = PredictionHeadTransform(cfg)

        # The output weights are the same as the input embeddings,
        # but there is　an output-only bias for each token.
        if cfg.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, (
                "bert_model_embedding_weights should not be None "
                "when setting --share_wd_cls_weight flag to be true"
            )
            assert cfg.hidden_size == bert_model_embedding_weights.size(1), (
                "hidden size has be the same as word embedding size when "
                "sharing word embedding weight and classifier weight"
            )
            self.decoder = nn.Linear(
                bert_model_embedding_weights.size(1),
                bert_model_embedding_weights.size(0),
                bias=False,
            )
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)
