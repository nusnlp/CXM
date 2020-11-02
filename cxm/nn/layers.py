from typing import List, Dict, Optional
import logging

import torch
import torch.nn.functional as F
from allennlp.common.from_params import FromParams
from allennlp.modules import FeedForward, Seq2SeqEncoder, TimeDistributed
from allennlp.nn.util import combine_tensors, masked_softmax, masked_max

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class GatedEncoding(torch.nn.Module, FromParams):
    """Gating over a sequence:
    * o_i = sigmoid(Wx_i) * x_i for x_i in X.
    """

    def __init__(self, gate: FeedForward):
        super(GatedEncoding, self).__init__()
        self.linear = gate  # put linear activation # nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            gated_x: batch * len * hdim
        """
        gate = self.linear(x.view(-1, x.size(2))).view(x.size())
        gate = torch.sigmoid(gate)
        gated_x = torch.mul(gate, x)
        return gated_x


class GatedMultifactorSelfAttnEnc(torch.nn.Module, FromParams):
    """Gated multi-factor self attentive encoding over a sequence:
    """

    def __init__(self, input_dim: int,
                 projector: Optional[FeedForward] = None,
                 gate: Optional[FeedForward] = None,
                 num_factor: int = 4):
        super(GatedMultifactorSelfAttnEnc, self).__init__()
        self.num_factor = num_factor
        if self.num_factor > 0:
            self.linear = projector
            total_dim = self.num_factor
        else:
            self.linear = None
            total_dim = 1
        assert gate is not None
        self.linear_gate = gate
        self._encoding_dim = input_dim
        self._merge_self_attention = TimeDistributed(torch.nn.Linear(
            self._encoding_dim * (total_dim + 1), self._encoding_dim))

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len
        Output:
            gated_selfmatched_x: batch * len * hdim
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        hdim = x.shape[2]
        if self.linear is not None:
            self_attn_multi = []
            y_multi = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1),
                                                              x.size(2) * self.num_factor)
            # y_multi = torch.relu(y_multi)
            y_multi = y_multi.view(x.size(0), x.size(1), x.size(2), self.num_factor)
            for fac in range(self.num_factor):
                y = y_multi.narrow(3, fac, 1).squeeze(-1)
                attn_fac = y.bmm(y.transpose(2, 1))
                attn_fac = attn_fac.unsqueeze(1)  # batch * 1 * len * len
                self_attn_multi.append(attn_fac)
            self_attn = torch.cat(self_attn_multi, 1)  # batch * num_factor * len * len
        else:
            self_attn = x.bmm(x.transpose(2, 1))  # batch * len * len
            self_attn = self_attn.unsqueeze(1)  # batch * 1 * len * len

        num_factors = self_attn.shape[1]  # m
        mask = x_mask.reshape(x_mask.size(0), x_mask.size(1), 1) \
                * x_mask.reshape(x_mask.size(0), 1, x_mask.size(1))  # batch * len * len

        self_mask = torch.eye(x_mask.size(1), x_mask.size(1), device=x_mask.device)
        self_mask = self_mask.reshape(1, x_mask.size(1), x_mask.size(1))
        mask *= (1 - self_mask)  # batch * len * len
        mask_multi = mask.unsqueeze(1).repeat(1, num_factors,
                                              1, 1)  # batch * num_factor * len * len

        # Normalize with softmax
        alpha = masked_softmax(self_attn, mask_multi, dim=-1)  # batch * num_factor * len * len

        # multifactor attentive enc
        multi_attn_enc = alpha.view(batch_size, num_factors * seq_len,
                                    -1).bmm(x).view(batch_size, num_factors,
                                                    seq_len, -1)  # batch * num_factors * len * hdim
        multi_attn_enc = multi_attn_enc.transpose(2, 1).contiguous().view(batch_size,
                                                             seq_len, -1)  # B * len * mH

        # merge with original x
        gate_input = [x]
        gate_input.append(multi_attn_enc)
        joint_ctx_input = torch.cat(gate_input, 2)  # B * len * 5H

        # gating
        gate_joint_ctx_self_match = self.linear_gate(joint_ctx_input.view(-1, joint_ctx_input.size(2))).view(
            joint_ctx_input.size())
        gate_joint_ctx_self_match = torch.sigmoid(gate_joint_ctx_self_match)

        gated_multi_attentive_enc = torch.mul(gate_joint_ctx_self_match, joint_ctx_input)
        merge_self_attn_vecs = torch.tanh(self._merge_self_attention(gated_multi_attentive_enc))

        return merge_self_attn_vecs


class AttnPooling(torch.nn.Module, FromParams):
    def __init__(self, projector: FeedForward,
                 intermediate_projector: FeedForward = None) -> None:
        super(AttnPooling, self).__init__()
        self._projector = projector
        self._int_proj = intermediate_projector

    def forward(self, xinit: torch.FloatTensor,
                xmask: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
        :param xinit: B * T * H
        :param xmask: B * T
        :return: B * H
        """
        if self._int_proj is not None:
            x = self._int_proj(xinit)
            x = x * xmask.unsqueeze(-1)
        else:
            x = xinit
        attn = self._projector(x)  # B * T * 1
        attn = attn.squeeze(-1)  # B * T
        attn = masked_softmax(attn, xmask, dim=-1)
        pooled = attn.unsqueeze(1).bmm(xinit).squeeze(1)  # B * H
        return pooled


