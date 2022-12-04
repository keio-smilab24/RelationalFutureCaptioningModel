import math

import torch
from torch import nn
import torchvision.models as models


class ImgEmbedder(nn.Module):
    def __init__(self):
        super(ImgEmbedder, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1024, 768)
        self.conv1 = nn.Conv2d(256, 456, 6, stride=2)
        self.conv2 = nn.Conv2d(456, 512, 5, stride=3)
        self.conv3 = nn.Conv2d(512, 768, 8, stride=1)

    def forward(self, x: torch.Tensor):
        B,L,_ = x.size()
        x = x.view(-1, 3, 224, 224)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x,(B,L,768))

        return x


class MultiModalEmbedding(nn.Module):
    """
    Construct the embeddings from word (+ video),
    position and token_type embeddings.
    input_ids (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, cfg, add_postion_embeddings=True):
        super().__init__()
        """
        add_postion_embeddings: whether to add absolute positional embeddings
        """
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.word_vec_size, padding_idx=0
        )
        
        self.word_fc = nn.Sequential(
            nn.LayerNorm(cfg.word_vec_size, eps=cfg.layer_norm_eps),
            nn.Dropout(cfg.hidden_dropout_prob),
            nn.Linear(cfg.word_vec_size, cfg.hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )
        self.img_embeddings = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(
                n_filters=cfg.hidden_size, max_len=cfg.max_position_embeddings * 2
            )
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)

        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """
        Note the from_pretrained does not work in-place, so you need to assign value to the embedding
        """
        assert (
            pretrained_embedding.shape == self.word_embeddings.weight.shape
        )  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=freeze,
            padding_idx=self.word_embeddings.padding_idx,
        )

    def forward(self, input_ids, img_feats, token_type_ids):
        """
        Args:
            input_ids: (N, L) | CLS, VID...VID, SEP BOS, W..W, EOS, PAD...PAD
            img_features: (N, L, D) | XX, VID..VID, XX...XX
            token_type_ids: (N, L, D)

        Returns:
        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        img_embeddings = self.img_embeddings(img_feats)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # words_embeddings += token_type_embeddings
        embeddings = words_embeddings + img_embeddings + token_type_embeddings

        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_filters, 2).float() * -(math.log(10000.0) / n_filters)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[: x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x