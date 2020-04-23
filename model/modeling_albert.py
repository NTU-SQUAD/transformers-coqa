from transformers import AlbertModel, AlbertPreTrainedModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
from .Layers import Multi_linear_layer
from .Layers import PoolerAnswerClass,PoolerEndLogits,PoolerStartLogits
import torch.nn as nn

class AlbertForConversationalQuestionAnswering(AlbertPreTrainedModel):
    def __init__(
            self,
            config,
            output_attentions=False,
            keep_multihead_output=False,
            n_layers=2,
            activation='relu',
            beta=100,
    ):
        super(AlbertForConversationalQuestionAnswering, self).__init__(config)
        self.output_attentions = output_attentions
        self.albert = AlbertModel(config)
        hidden_size = config.hidden_size
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)
        self.start_n_top = 2 #config.start_n_top
        self.end_n_top = 2 #config.end_n_top
        self.rational_l = Multi_linear_layer(n_layers, hidden_size,
                                             hidden_size, 1, activation)
        self.logits_l = Multi_linear_layer(n_layers, hidden_size, hidden_size,
                                           2, activation)
        self.unk_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 1,
                                        activation)
        self.attention_l = Multi_linear_layer(n_layers, hidden_size,
                                              hidden_size, 1, activation)
        self.yn_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 2,
                                       activation)
        self.beta = beta

        self.init_weights()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx = None,
            head_mask=None,
            p_mask = None,
            cls_index_pos=None,
            span_is_impossible=None
    ):

        outputs = self.albert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs
        else:
            final_hidden, pooled_output = outputs
        # rational_logits = self.rational_l(final_hidden)
        # rational_logits = torch.sigmoid(rational_logits)
        #
        # final_hidden = final_hidden * rational_logits
        #
        # logits = self.logits_l(final_hidden)
        #
        # start_logits, end_logits = logits.split(1, dim=-1)
        #
        # start_logits, end_logits = start_logits.squeeze(
        #     -1), end_logits.squeeze(-1)
        #
        # segment_mask = token_type_ids.type(final_hidden.dtype)
        #
        # rational_logits = rational_logits.squeeze(-1) * segment_mask
        #
        # start_logits = start_logits * rational_logits
        #
        # end_logits = end_logits * rational_logits
        #
        # unk_logits = self.unk_l(pooled_output)
        #
        # attention = self.attention_l(final_hidden).squeeze(-1)
        #
        # attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        #
        # attention = F.softmax(attention, dim=-1)
        #
        # attention_pooled_output = (attention.unsqueeze(-1) *
        #                            final_hidden).sum(dim=-2)
        #
        # yn_logits = self.yn_l(attention_pooled_output)
        #
        # yes_logits, no_logits = yn_logits.split(1, dim=-1)
        #
        # start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        # end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        #
        # new_start_logits = torch.cat(
        #     (yes_logits, no_logits, unk_logits, start_logits), dim=-1)
        # new_end_logits = torch.cat(
        #     (yes_logits, no_logits, unk_logits, end_logits), dim=-1)
        #
        # if start_positions is not None and end_positions is not None:
        #
        #     start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
        #
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = new_start_logits.size(1)
        #     start_positions.clamp_(0, ignored_index)
        #     end_positions.clamp_(0, ignored_index)
        #
        #     span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #
        #     start_loss = span_loss_fct(new_start_logits, start_positions)
        #     end_loss = span_loss_fct(new_end_logits, end_positions)
        #
        #     # rational part
        #     alpha = 0.25
        #     gamma = 2.
        #     rational_mask = rational_mask.type(final_hidden.dtype)
        #
        #     rational_loss = -alpha * ((1 - rational_logits)**gamma) * rational_mask * torch.log(rational_logits + 1e-7) \
        #                     - (1 - alpha) * (rational_logits**gamma) * (1 - rational_mask) * \
        #                     torch.log(1 - rational_logits + 1e-7)
        #
        #     rational_loss = (rational_loss * segment_mask).sum() / segment_mask.sum()
        #
        #     assert not torch.isnan(rational_loss)
        #
        #     total_loss = (start_loss + end_loss) / 2 + rational_loss * self.beta
        #     return total_loss
        #
        # return start_logits, end_logits, yes_logits, no_logits, unk_logits


        hidden_states = final_hidden
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)


        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index_pos, span_is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index_pos is not None and span_is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_idx)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, span_is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5
            outputs =total_loss

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_idx
            )  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs