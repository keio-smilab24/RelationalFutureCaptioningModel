"""
MART loss.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing/(tgt_vocab_size-1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = (
            target != self.ignore_index
        )  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")

class CLIPloss(nn.Module):
    """
    CLIPで用いられているloss
    https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
    """
    def __init__(
        self,
        hidden_dim: int=768,
        max_t_length: int=22,):
        super().__init__()
        self.w = nn.Linear(max_t_length * hidden_dim, hidden_dim)
        self.t = torch.randn(1, requires_grad=True).cuda()
        self.i_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.t_loss = nn.CrossEntropyLoss(ignore_index=1)
        self.norm_i = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

    def forward(self, clip, text):
        text = torch.flatten(text, 1)
        text = self.w(text)
        i_e = self.norm_i(clip)
        t_e = self.norm_t(text)
        logits = torch.matmul(i_e, torch.t(t_e)) * torch.exp(self.t)
        n = i_e.shape[0]
        labels = torch.arange(n, device=torch.device("cuda"))
        loss_i = self.i_loss(logits, labels)
        loss_t = self.t_loss(logits, labels)
        cliploss = (loss_i + loss_t) / 2
        return cliploss