# Date: 2020/12/3
# Author: Qianqian Peng

import torch
from mention_detection.mention_detection import MentionScoresHead,MentionLoss
from blink_biencoder import wiki_encoder
import torch.nn as nn
# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )
def load_model(model_name, model_path):
    model_config = BertConfig_new.from_pretrained(model_name)
    bert_model = BertModel_new.from_pretrained(model_path, config=model_config)
    return bert_model


from mention_detection.mention_data_proc import ReadTrainDectMent, IterData
from BiencoderRanker2 import predict
from mention_detection.utils import *
from transformers import BertModel as BertModel_new
from transformers import BertConfig as BertConfig_new
class Test(nn.Module):
    def __init__(self,base_model_name='bert-base-uncased',base_model_path='./model/bert-base-uncased',bert_output_dim=768,score_method='qa_linear',max_mention_length=10):
        super(Test,self).__init__()

        self.bert_output_dim = bert_output_dim
        self.score_method = score_method
        self.max_mention_length = max_mention_length

        self.mention_score = MentionScoresHead(self.bert_output_dim, self.score_method, self.max_mention_length)
        self.mention_loss = MentionLoss()
        self.ques_encoder = load_model(base_model_name, base_model_path)
    def forward(self,input_ids_ctxt, input_mask_ctxt,gold_mention_bounds, gold_mention_bounds_mask):
        input_ids = torch.LongTensor(input_ids_ctxt).to('cuda')
        input_mask = torch.tensor(input_mask_ctxt).to('cuda')
        input_mask_bool = input_mask.bool().to('cuda')
        token_type_ids = torch.ones(input_ids.size(), dtype=torch.int8).to('cuda')
        gold_mention_bounds = torch.clone(torch.tensor(gold_mention_bounds)).to('cuda')
        gold_mention_bounds_mask_bool = torch.tensor(gold_mention_bounds_mask).bool().to('cuda')
        bert_output, _ = self.ques_encoder(input_ids, token_type_ids, input_mask)
        mention_scores, mention_bounds = self.mention_score(bert_output,
                                                            input_mask_bool)  # forward(self, bert_output, input_mask)
        loss = self.mention_loss(gold_mention_bounds, gold_mention_bounds_mask_bool, mention_scores,
                                 mention_bounds)  # f
        return loss,mention_scores,mention_bounds
import torch.optim as optim

LR=0.02
batch_size=128
num_epoch=10
eval_n=2
top_k=1

model=Test().to('cuda')

optimizer=optim.SGD(model.parameters(),lr=LR)

trainfile_path='./Data/train_merge.jsonl'
train_data=ReadTrainDectMent(trainfile_path,True)
train_data.padding()

num_batch=train_data.data_size//batch_size

for epoch in range(num_epoch):
    n = 1
    dataload = IterData(train_data, batch_size, True, True)
    for input_ids_ctxt, input_mask_ctxt, gold_mention_bounds, gold_mention_bounds_mask, gold_entity_local_id in dataload:
        optimizer.zero_grad()

        loss,scores,bounds=model(input_ids_ctxt , input_mask_ctxt,gold_mention_bounds, gold_mention_bounds_mask)

        if n % eval_n == 0:
            top_k_bounds = predict(scores, bounds, top_k)
            pres, rec = compute_precise_recall_overlap2(top_k_bounds, torch.tensor(gold_mention_bounds))
            print('*' * 25, 'epoch: ', epoch + 1, '/', num_epoch, '\t', 'batch: ', n, '/', num_batch, '*' * 25)
            print('pres: ', pres)
            print('recall: ', rec)
            print('ment_loss: ', loss)
            print('\n')
        n += 1
        loss.backward()
        optimizer.step()