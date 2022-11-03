import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, RobertaModel


class DyReF(BertPreTrainedModel):

    def __init__(self, config, aggregation="mean", masking="full"):
        super().__init__(config)

        self.encoder_name = config.model_type
        if "roberta" in self.encoder_name:
            self.bert = RobertaModel(config)
        else:
            self.bert = BertModel(config)

        # the queries
        self.prefixes = nn.Parameter(data=torch.randn(2, config.hidden_size) * 0.02)

        self.aggregation = aggregation

        if self.aggregation == "weighted":
            self.att_ag = nn.Sequential(
                nn.Linear(config.hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        self.masking = masking

        self.init_weights()

    def mask_no_modify_input(self, new_att):
        new_att = new_att.unsqueeze(-2)
        size = new_att.size(-1)
        eye = torch.ones((size, size))

        # input cannot attend to queries
        eye[2:, :2] = 0

        # queries cannot attend to each other
        if self.masking == "independent":
            eye[0, 1] = 0
            eye[1, 0] = 0
        # start query cannot attend to end query
        elif self.masking == "causal":
            eye[0, 1] = 0
        elif self.masking == "bidirectionnal":
            eye = eye
        else:
            raise ValueError

        eye = eye.unsqueeze(0).to(new_att.device)

        return eye * new_att

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None):

        B, L = input_ids.size()

        # word embedding
        word_embeddings = self.bert.embeddings.word_embeddings.forward(input_ids)

        # repeat query for every batches "dynamic !!'
        queries = self.prefixes.unsqueeze(0).repeat(B, 1, 1)

        # concat queries and word embeddings
        input_embeddings = torch.cat([queries, word_embeddings], dim=1)

        # modify attention masks
        new_mask = self.extends_attention_mask(attention_mask)

        if self.masking != "full":
            new_mask = self.mask_no_modify_input(new_mask)

        # compute contextual
        bert_out = self.bert(inputs_embeds=input_embeddings,
                             attention_mask=new_mask,
                             token_type_ids=self.extends_token_types(token_type_ids))

        all_hidden = torch.stack(bert_out[-1][1:])  # N, B, L, D

        if self.aggregation == "mean":
            queries = all_hidden[:, :, :2, :].mean(0).transpose(0, 1)
        elif self.aggregation == "last":
            queries = all_hidden[-1, :, :2, :].transpose(0, 1)
        elif self.aggregation == "max":
            queries = all_hidden[:, :, :2, :].max(0).values.transpose(0, 1)
        elif self.aggregation == "weighted":
            queries = all_hidden[:, :, :2, :]  # N, B, 2, D
            token_score = self.att_ag(queries)  # N, B, 2, 1
            queries = (queries * token_score).sum(0).transpose(0, 1)

        # remove queries to get sequence representation
        x = bert_out[0][:, 2:, :]  # B, L, D

        # start/end logits
        start_logits, end_logits = torch.einsum('cbd,bld->cbl', queries, x)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0

        outputs = (start_logits, end_logits)

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.long())
            end_loss = loss_fct(end_logits, end_positions.long())

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs

    def extends_attention_mask(self, attention_mask):

        if attention_mask is None:
            return attention_mask

        B, L = attention_mask.size()
        new_att = torch.ones(B, L + 2).type_as(attention_mask)
        new_att[:, 2:] = attention_mask

        return new_att

    def extends_token_types(self, token_type_ids):

        if token_type_ids is None:
            return token_type_ids

        B, L = token_type_ids.size()
        new_tok = torch.zeros(B, L + 2).type_as(token_type_ids)
        new_tok[:, 2:] = token_type_ids

        return new_tok
