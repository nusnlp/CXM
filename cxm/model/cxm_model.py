import logging
from typing import Any, Dict, List
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, Auc, F1Measure, MeanAbsoluteError
from allennlp.nn.util import masked_max, masked_softmax, get_text_field_mask

from cxm.nn.layers import GatedMultifactorSelfAttnEnc, AttnPooling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("cxm_model")
class CXMModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 qdep_henc_rnn: Seq2SeqEncoder,
                 senc_self_attn: GatedMultifactorSelfAttnEnc,
                 attnpool: AttnPooling,
                 output_ffl: FeedForward,
                 initializer: InitializerApplicator,
                 dropout: float = 0.3,
                 ) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

        self._qdep_henc_rnn = qdep_henc_rnn
        self._senc_self_attn = senc_self_attn

        self._variational_dropout = InputVariationalDropout(dropout)
        self._attn_pool = attnpool
        self._output_ffl = output_ffl

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._mae = MeanAbsoluteError()
        self._loss = torch.nn.MSELoss()
        self._softmax = torch.nn.Softmax(dim=1)
        initializer(self)

    def forward(self,
                combined_source: Dict[str, torch.LongTensor],
                prev_turns: Dict[str, torch.LongTensor],
                curr_utt: Dict[str, torch.LongTensor],
                prev_turns_mask: torch.FloatTensor,
                utt_mask: torch.FloatTensor,
                label_scores: torch.FloatTensor = None,
                label_gold: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        
        embedded_source = self._text_field_embedder(combined_source)
        source_mask = get_text_field_mask(combined_source)
        embedded_source = self._variational_dropout(embedded_source)




        embedded_history = embedded_source * prev_turns_mask.unsqueeze(-1)
        embedded_question = embedded_source * utt_mask.unsqueeze(-1)

        scores = embedded_question.bmm(embedded_history.transpose(2, 1))
        mask = utt_mask.unsqueeze(1) * prev_turns_mask.unsqueeze(-1)
        alpha = masked_softmax(scores, mask, dim=-1)
        qdep_hist = alpha.bmm(embedded_history)

        x = torch.cat([embedded_question, qdep_hist], -1)
        x = self._qdep_henc_rnn(x, source_mask)
        x = x * utt_mask.unsqueeze(-1).float()

        x = self._senc_self_attn(x, utt_mask)
        x = self._variational_dropout(x)

        cls_tokens = self._attn_pool(x, utt_mask)

        pred_label_scores = self._output_ffl(cls_tokens)

        output = self._softmax(pred_label_scores)

        _, pred_label = output.max(1)

        assert output.shape[0] == pred_label.shape[0]

        output_dict = {'label_logits': pred_label_scores,
                       'label_probs': output,
                       'label_pred': pred_label,
                       'metadata': metadata}
        
        if label_scores is not None:
            scores = label_scores
            label = label_gold.long()
            loss = self._loss(output, scores)
            self._accuracy(output, label)

            output_dict['loss'] = loss


        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }