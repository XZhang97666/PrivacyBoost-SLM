import copy
import math
import os
import warnings
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import  BertOnlyMLMHead

from transformers.utils.generic import ModelOutput
from transformers import BertModel


class BertForMTLModelOutput(ModelOutput):
    def __init__(
        self,
        loss=None,
        logits=None,
        mc_logits=None,
        mlm_logits=None,
        hidden_states=None,
        attentions=None,
    ):
        self.loss = loss
        self.logits = logits
        self.mc_logits = mc_logits
        self.mlm_logits = mlm_logits
        self.hidden_states = hidden_states
        self.attentions = attentions


class BertForMTL(BertPreTrainedModel):
    def __init__(self, config,args):
        super().__init__(config)
        self.bert = BertModel(config)
        self.smoothing_ratio=args.smoothing_ratio
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)


        #MLM
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask= None,
        inputs_embeds = None,
        mc_label = None,
        mlm_labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        mt_weight=None) :
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #MC
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        #MLM
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

 
        loss=None
        if mt_weight is not None:               
            loss_mc = None
            if mc_label is not None:
                loss_mc = CrossEntropyLoss()(reshaped_logits, mc_label)
            masked_lm_loss = None
            if mlm_labels is not None:
                masked_lm_loss = CrossEntropyLoss()(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1)) 

            loss=mt_weight*loss_mc+(1-mt_weight)*masked_lm_loss
        else:
            if mc_label is not None:
                loss = CrossEntropyLoss(label_smoothing=self.smoothing_ratio)(reshaped_logits, mc_label)


        if not return_dict:
            output = (reshaped_logits, prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output



        return BertForMTLModelOutput(
            loss=loss,
            logits=None,
            mc_logits=reshaped_logits,
            mlm_logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



