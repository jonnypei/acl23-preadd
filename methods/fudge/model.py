import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel

from methods.fudge.constants import *


class Model(nn.Module):
    def __init__(self, args, gpt_pad_id):
        super(Model, self).__init__()

        self.toxicity = (args.task == "toxicity")
        self.bias = (args.task == "bias")

        if self.toxicity or self.bias:
            print("Base Model:", args.model_string)
            self.model = AutoModel.from_pretrained(args.model_string)
            if args.model_string == 'facebook/opt-125m':
                proj_size = self.model.config.word_embed_proj_dim
            elif args.model_string == 'EleutherAI/gpt-neo-125M':
                proj_size = self.model.config.hidden_size
            self.classification_head = nn.Linear(proj_size, 1)
        else:
            raise NotImplementedError  # Can add/refactor different models in the future

    def forward(
        self,
        inputs: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            inputs,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classification_head(outputs.last_hidden_state)

        return logits
