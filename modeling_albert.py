from transformers import AlbertModel, AlbertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch

class Multi_linear_layer(nn.Module):
    def __init__(self,
                 n_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 activation=None):
        super(Multi_linear_layer, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, n_layers - 1):
            self.linears.append(nn.Linear(hidden_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))
        self.activation = getattr(F, activation)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        linear = self.linears[-1]
        x = linear(x)
        return x

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

        rational_logits = self.rational_l(final_hidden)
        rational_logits = torch.sigmoid(rational_logits)

        final_hidden = final_hidden * rational_logits

        logits = self.logits_l(final_hidden)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits, end_logits = start_logits.squeeze(
            -1), end_logits.squeeze(-1)

        segment_mask = token_type_ids.type(final_hidden.dtype)

        rational_logits = rational_logits.squeeze(-1) * segment_mask

        start_logits = start_logits * rational_logits

        end_logits = end_logits * rational_logits

        unk_logits = self.unk_l(pooled_output)

        attention = self.attention_l(final_hidden).squeeze(-1)

        attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        attention = F.softmax(attention, dim=-1)

        attention_pooled_output = (attention.unsqueeze(-1) *
                                   final_hidden).sum(dim=-2)

        yn_logits = self.yn_l(attention_pooled_output)

        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        new_start_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, start_logits), dim=-1)
        new_end_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, end_logits), dim=-1)

        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # rational part
            alpha = 0.25
            gamma = 2.
            rational_mask = rational_mask.type(final_hidden.dtype)

            rational_loss = -alpha * ((1 - rational_logits)**gamma) * rational_mask * torch.log(rational_logits + 1e-7) \
                            - (1 - alpha) * (rational_logits**gamma) * (1 - rational_mask) * \
                            torch.log(1 - rational_logits + 1e-7)

            rational_loss = (rational_loss * segment_mask).sum() / segment_mask.sum()

            assert not torch.isnan(rational_loss)

            total_loss = (start_loss + end_loss) / 2 + rational_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits