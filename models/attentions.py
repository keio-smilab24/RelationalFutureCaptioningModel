import math

import torch
import torch.nn as nn

from models.misc import SelfOutput


class MHSA(nn.Module):
    """
    TransformerにおけるMHA
    """

    def __init__(self, cfg):
        super().__init__()
        self.self = SelfAttention(cfg)
        self.output = SelfOutput(cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x, attention_mask=None, clip_his=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)
        Returns:
        """
        # x = self.layernorm(x)
        if clip_his is not None:
            self_output = self.self(clip_his, x, x, attention_mask)
        else:
            self_output = self.self(x, x, x, attention_mask)
        att = self.output(self_output, x)
        return att


class SelfAttention(nn.Module):
    """
    Attentionの計算
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.hidden_size, cfg.num_attention_heads)
            )
        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = int(cfg.hidden_size / cfg.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_w = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key_w = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value_w = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
        """
        # only need to mask the dimension where the softmax
        # (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        if attention_mask is not None:
            attention_mask = (
                1 - attention_mask.unsqueeze(1)
            ) * -10000.0  # (N, 1, Lq, L)
        mixed_query_layer = self.query_w(query)
        mixed_key_layer = self.key_w(key)
        mixed_value_layer = self.value_w(value)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        att_w = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        att_w = att_w / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers
            # in BertModel forward() function)
            att_w = att_w + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(att_w)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class RelationalSelfAttention(nn.Module):
    """
    Relational self-attention (RSA)
    https://arxiv.org/pdf/2111.01673.pdf
    """
    def __init__(self, cfg, m=3):
        super().__init__()
        self.cfg = cfg
        self.m = m
        self.hidden_size = cfg.hidden_size
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.p = torch.randn((m, self.hidden_size), requires_grad=True).cuda()
        self.h = torch.randn((m * self.hidden_size, m), requires_grad=True).cuda()
        self.g = torch.randn((m, self.hidden_size), requires_grad=True).cuda()
        self.one = torch.ones((m, 1)).cuda()

    def forward(self, target, cont):
        query = self.query_layer(target).reshape(-1, self.hidden_size, 1)
        key = self.key_layer(cont)
        value = self.value_layer(cont)

        # basic kernel
        kernel_v = torch.matmul(self.p, query).reshape(-1, 1, self.m)

        # relational kernel
        q = torch.matmul(self.one, torch.transpose(query, 1, 2))
        x_q = torch.mul(q, key)
        x_q = x_q.reshape((-1, 1, self.m * self.hidden_size))
        kernel_r = torch.matmul(x_q, self.h).reshape(-1, 1, self.m)
        kernel = kernel_v + kernel_r

        # basic context
        # basic_cont = context.clone()

        # relational context
        xg = value.clone()
        xg = torch.transpose(xg, 1, 2)
        _xg = torch.matmul(xg, self.g)
        x_nr = torch.matmul(value, _xg)
        context = x_nr + value

        output = torch.matmul(kernel, context).reshape(-1, self.hidden_size)

        return output


class MultiHeadRSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_image = cfg.max_v_len - 2
        self.hidden_size = cfg.hidden_size
        self.head = cfg.max_v_len - 2
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.p = torch.randn((self.head, self.num_image, self.hidden_size), requires_grad=True).cuda()
        self.h = torch.randn((self.head, self.num_image * self.hidden_size, self.num_image), requires_grad=True).cuda()
        self.g = torch.randn((self.head, self.num_image, self.hidden_size), requires_grad=True).cuda()
        self.one = torch.ones((self.num_image, 1)).cuda()

    def forward(self, target, cont):
        query = self.query_layer(target)
        key = self.key_layer(cont).reshape(-1, 1, self.head, self.hidden_size)
        key = key.repeat((1,self.head,1,1))
        value = self.value_layer(cont).reshape(-1, 1, self.head, self.hidden_size)
        value = value.repeat((1,self.head,1,1))

        query = query.reshape((-1, 1, self.head, self.hidden_size)).permute(0, 2, 1, 3)
        key = key.reshape((-1, self.num_image, self.head, self.hidden_size)).permute(0, 2, 1, 3)
        value = value.reshape((-1, self.num_image, self.head, self.hidden_size)).permute(0, 2, 1, 3)

        # basic kernel
        kernel_v = torch.matmul(self.p, query.permute(0, 1, 3, 2)).reshape(-1, self.head, 1, self.num_image)

        # relational kernel
        q = torch.matmul(self.one, query)
        x_q = torch.mul(q, key)
        x_q = x_q.reshape((-1, self.head, 1, self.num_image * self.hidden_size))
        kernel_r = torch.matmul(x_q, self.h).reshape(-1, self.head, 1, self.num_image)
        kernel = kernel_v + kernel_r

        # relational context
        xg = value.clone()
        xg = torch.transpose(xg, 2, 3)
        _xg = torch.matmul(xg, self.g)
        x_nr = torch.matmul(value, _xg)
        context = x_nr + value

        output = torch.matmul(kernel, context).reshape(-1, self.head, self.hidden_size)
        return output