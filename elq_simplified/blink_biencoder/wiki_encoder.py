# Date: 2020/11/28
# Author: Qianqian Peng

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from common.bert_base import BertEncoder
from torch import nn
import json

def load_para(para_path):
    file=open(para_path,'r')
    para=file.readline()
    file.close()
    return json.loads(para)


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class WikiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(WikiEncoderModule, self).__init__()
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = cand_bert.config

    def forward(
        self,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_cands



# para=load_para('../model/biencoder/wiki_encoder_large.json')
# model = WikiEncoderModule(para)
# # state_dict=torch.load('../model/biencoder/wiki_encoder_large.bin')
#
# model.load_state_dict(torch.load('../model/biencoder/wiki_encoder_large.bin'))


#save:             bert_model.embeddings.word_embeddings.weight
#cur: cand_encoder.bert_model.embeddings.word_embeddings.weight

# state_dict=list(state_dict.items())
# for i in range(len(state_dict)):
#     cur=state_dict.pop(0)
#     key='cand_encoder.'+cur[0]
#     state_dict.append((key,cur[1]))
#
# import collections
#
# state_dict=collections.OrderedDict(state_dict)
#
#
# model.load_state_dict(state_dict)
#
# torch.save(model.state_dict(),'F:/PYProject/elq3/model/biencoder/wiki_encoder_large.bin')
