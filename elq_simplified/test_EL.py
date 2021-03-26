# Date: 2020/12/4
# Author: Qianqian Peng

from mention_detection.mention_detection import load_model

import torch
from transformers import BertTokenizer as BertTokenizer_new
from transformers import BertConfig as BertConfig_new
from transformers import BertModel as BertModel_new
import torch.nn as nn

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

bert_new=BertModel_new.from_pretrained('./model/bert-large-uncased',config=BertConfig_new.from_pretrained('bert-large-uncased'))
bert_old=BertModel.from_pretrained('./model/bert-large-uncased',config=BertConfig.from_pretrained('bert-large-uncased'))