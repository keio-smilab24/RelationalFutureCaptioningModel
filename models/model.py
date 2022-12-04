import logging

import torch
from torch import nn

from losses.loss import LabelSmoothingLoss
from utils.utils import count_parameters
from losses.loss import CLIPloss
from utils.configs import Config
from models.embedder import ImgEmbedder, MultiModalEmbedding
from models.encoder import RSAEncoder, TransformerEncoder
from models.decoder import TransformerDecoder, PredictionHead


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACTION_WEIGHT = {111: 131, 94: 628} 

# # default infinity (cfg.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float("inf")


class RecursiveTransformer(nn.Module):
    def __init__(
        self,
        cfg: Config,
        vocab_size: int
    ):
        super().__init__()
        self.cfg = cfg
        self.cfg.vocab_size = vocab_size
        
        self.img_embedder = ImgEmbedder()
        self.embeddings = MultiModalEmbedding(cfg, add_postion_embeddings=True)
        self.TSModule = RSAEncoder(cfg)
        self.encoder = TransformerEncoder(cfg)
        
        decoder_classifier_weight = (
            self.embeddings.word_embeddings.weight
            if self.cfg.share_wd_cls_weight
            else None
        )
        
        self.transformerdecoder = TransformerDecoder(cfg)
        self.decoder = PredictionHead(cfg, decoder_classifier_weight)
        
        if self.cfg.label_smoothing != 0:
            self.loss_func = LabelSmoothingLoss(
                cfg.label_smoothing, cfg.vocab_size, ignore_index=-1
            )
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.contloss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.actionloss_func = nn.CrossEntropyLoss()
        
        # clipの特徴量の次元
        input_size = cfg.clip_dim
        # TODO : memo cnnを使ったadjustがbetterな気がする
        self.size_adjust = nn.Linear(150528, 768)
        self.upsampling = nn.Linear(768, 1024)
        self.pred_f = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
        )
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Dropout(0.2),
        )

        self.rec_loss = nn.MSELoss()
        self.apply(self.init_bert_weights)
        self.cliploss = CLIPloss(hidden_dim=cfg.hidden_size, max_t_length=cfg.max_t_len)

        self.idx = 0

    def init_bert_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(
        self,
        input_ids: torch.Tensor,
        features: torch.Tensor,
        input_masks: torch.Tensor,
        token_type_ids: torch.Tensor
    ):
        """
            singleure step forward in the recursive struct
        Args:
            input_ids (torch.Tensor):
            video_features (torch.Tensor):
            input_masks (torch.Tensor):
            token_type_ids (torch.Tensor):
        """
        # 画像特徴だけ抽出
        img_feats = features[:, 1:self.cfg.max_v_len-1, :].clone()  # (B, 7, 150528)
        features = self.size_adjust(features)                       # (B, 31, 150528) -> (B, 31, 768)
        
        self.pred_reconst = []
        self.gt_rec = []
        
        # resnetを用いた特徴量抽出
        features[:, 1:self.cfg.max_v_len-1, :] = self.img_embedder(img_feats)

        # 再構成用
        rec_feature = features[:,1,:].clone()
        # 画像特徴量only
        img_feats = features[:, 1:self.cfg.max_v_len-1, :].clone()

        # Time Series Module
        _, img_feats = self.TSModule(img_feats)

        embeddings = self.embeddings(input_ids, features, token_type_ids)
        encoded_layer_outputs = self.encoder(
            embeddings, input_masks, output_all_encoded_layers=False
        )
        decoded_layer_outputs = self.transformerdecoder(
            encoded_layer_outputs[-1], input_masks, img_feats
        )
        prediction_scores = self.decoder(
            decoded_layer_outputs[-1]
        )  # (N, L, vocab_size)
        return encoded_layer_outputs, prediction_scores, rec_feature

    def forward(
        self,
        input_ids_list,
        video_features_list,
        input_masks_list,
        token_type_ids_list,
        input_labels_list,
        gt_rec=None,
    ):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits,
                `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,

        Returns:
        """
        step_size = len(input_ids_list) # 1
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        pred_reconst = []
        gt_reconst = []
        action_score = []

        if gt_rec is not None:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores, pred_tmp = self.forward_step(
                    input_ids_list[idx],
                    video_features_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx]
                )
                gt_reconst.append(gt_rec[idx])
                pred_reconst.append(pred_tmp)
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 7, :])
        else:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores = self.forward_step(
                    input_ids_list[idx],
                    video_features_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx]
                )
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 7, :])
        
        # compute loss, get predicted words
        caption_loss = 0.0
        for idx in range(step_size):
            snt_loss = self.loss_func(
                prediction_scores_list[idx].view(-1, self.cfg.vocab_size),
                input_labels_list[idx].view(-1),
            )
            """
            # TODO : 確認 ここの7という値はなに
            """
            gt_action_list = input_labels_list[idx][:, 7]
            act_score_list = action_score[idx].cpu()
            iwp_loss = 0.0
            for actidx in range(len(gt_action_list)):
                gt_action = torch.tensor([gt_action_list[actidx]], dtype=int)
                gt_idx = gt_action.tolist()
                if gt_idx[0] == -1:
                    continue
                if gt_idx[0] in ACTION_WEIGHT:
                    iwp_loss += (1 / ACTION_WEIGHT[gt_idx[0]]) * self.actionloss_func(act_score_list[actidx].view(-1, self.cfg.vocab_size), gt_action)
                else:
                    iwp_loss += (1 / 300) * self.actionloss_func(act_score_list[actidx].view(-1, self.cfg.vocab_size), gt_action)
            clip_loss = 0.0

            clip_loss += self.cliploss(pred_reconst[idx], encoded_outputs_list[idx][0][:, self.cfg.max_v_len:, :])
            if gt_rec is not None:
                rec_loss = self.rec_loss(pred_reconst[idx].reshape(-1, 16, 16, 3), gt_reconst[idx] / 255.)

            caption_loss += 15 * snt_loss + 500 * rec_loss + 4 * clip_loss + iwp_loss / 100
        caption_loss /= step_size
        return caption_loss, prediction_scores_list, snt_loss, rec_loss, clip_loss


def create_model(
    cfg: Config,
    vocab_size: int,
    verbose: bool = True,
) -> nn.Module:
    """
    Args:
        cfg: config
        vocab_size: len(train_set.word2idx).
        verbose: Print model name and number of parameters.
    """
    cfg.max_position_embeddings = cfg.max_v_len + cfg.max_t_len
    cfg.vocab_size = vocab_size

    model = RecursiveTransformer(cfg, vocab_size=vocab_size)

    # output model properties
    if verbose: # true
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model)
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            count_parameters(model.embeddings.word_embeddings)

    return model