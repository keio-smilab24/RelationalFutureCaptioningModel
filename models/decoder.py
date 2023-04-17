import torch
from torch import nn
import torch.nn.functional as F

from models.attentions import MultiHeadAttention
from models.misc import FeedforwardNeuralNetModel, make_pad_shifted_mask


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm3 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm4 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm5 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm6 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.LayerNorm7 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # attention
        self.attention = MultiHeadAttention(cfg)
        self.attention2 = MultiHeadAttention(cfg)
        self.attention3 = MultiHeadAttention(cfg)
        self.attention4 = MultiHeadAttention(cfg)
        self.attention5 = MultiHeadAttention(cfg)
        self.attention6 = MultiHeadAttention(cfg)
        self.attention7 = MultiHeadAttention(cfg)

        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.rand2 = torch.randn(1, requires_grad=True).cuda()
        self.rand3 = torch.randn(1, requires_grad=True).cuda()
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout3 = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout4 = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout5 = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout6 = nn.Dropout(cfg.hidden_dropout_prob)
        self.dropout7 = nn.Dropout(cfg.hidden_dropout_prob)


        # ffn
        self.ffn = FeedforwardNeuralNetModel(cfg.hidden_size, cfg.hidden_size * 2, cfg.hidden_size)
        self.rand_z = torch.randn(1, requires_grad=True).cuda()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_his: torch.Tensor,
        make_knn_dstore: bool=False
        ):
        # attention layer
        # first layer
        q1 = x
        kv1 = clip_his

        x_text1 = self.attention(x=q1, source_kv = kv1)
        x_text1 = self.dropout(x_text1)
        x_text1 = x_text1 + q1
        x_text1= self.LayerNorm(x_text1)
        
        q2 = clip_his
        kv2 = x

        x_img1 = self.attention2(x=q2, source_kv=kv2)
        x_img1 = self.dropout2(x_img1)
        x_img1 = x_img1 + q2
        x_img1 = self.LayerNorm2(x_img1)

        # secoind layer
        q3 = x_text1
        kv3 = x_img1

        x_text2 = self.attention3(x=q3, source_kv=kv3)
        x_text2 = self.dropout3(x_text2)
        x_text2 = x_text2 + q3
        x_text2 = self.LayerNorm3(x_text2)

        q4 = x_img1
        kv4 = x_text1

        x_img2 = self.attention4(x=q4, source_kv=kv4)
        x_img2 = self.dropout4(x_img2)
        x_img2 = x_img2 + q4
        x_img2 = self.LayerNorm4(x_img2)
        
        # tird layer 
        q5 = x_text2
        kv5 = x_img2

        x_text3 = self.attention3(x=q5, source_kv=kv5)
        x_text3 = self.dropout5(x_text3)
        x_text3 = x_text3 + q5
        x_text3 = self.LayerNorm5(x_text3)

        q6 = x_img2
        kv6 = x_text2

        x_img3 = self.attention3(x=q6, source_kv=kv6)
        x_img3 = self.dropout6(x_img3)
        x_img3 = x_img3 + q6
        x_img3 = self.LayerNorm6(x_img3)
        

        # final attention layer
        q7 = x_text3
        kv7 = x_img3

        x_final = self.attention3(x=q7, source_kv=kv7)
        x_final = self.dropout7(x_final)
        x_final = x_final + q7
        x_final = self.LayerNorm7(x_final)

        if make_knn_dstore: 
            knn_feat = x_final.clone().detach().cpu()

        # ffn
        identity_x = x_final.clone().cuda()
        x_final = self.ffn(x_final)
        output = self.rand_z*identity_x + (1 - self.rand_z)*x_final
        output = self.LayerNorm(output)

        if make_knn_dstore:
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
