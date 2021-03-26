# Date: 2020/11/28
# Author: Qianqian Peng

##################################################version-4#############################################################
# Modification:
# 2020/12/5--0:21
# 准备将不正确的mention bounds在entity linking的步骤中去掉。因为模型学习完后，在测试EL的precision时，为0。
# 即使是不冻结wiki encoder的参数，最后还是为0，所以我怀疑正确的信息太少，而错误的信息为正确信息的百倍，所以
# 准备在EL的过程中去掉不正确的mention bounds试试。（先睡觉。。。）
#
#Modification:
# 2020/12/5--16:18
#修改完成
#
#Modification:
#2020/12/5--19:00
#增加变换bert-base-uncased的输出维度与large的相同的方法的复杂度
#
#Modification:
#2020/12/5--19:27
#将mention_detection的bert换成了large版本
#因为我发现facebook的elq用的large，这样就不需要变换维度
#
#遇到的问题：
#换成bert-large遇到的问题是，单独做mention detection可以有很好的的precision，而加入EL后，不仅mention detection做不出来，EL也没有效果
#无论是将learning rate调大还是调小，batch size调大还是调小，都没有用。反而在用bert-base做mention detection加EL时，mention detection
#的precision能够达到0.95以上，虽然EL的precision为0。
##################################################version-4#############################################################

import torch
from mention_detection.mention_detection import MentionScoresHead,MentionLoss,load_model

from blink_biencoder import wiki_encoder
import torch.nn as nn
from common.bert_base import BertEncoder
# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )



class BiencoderRanker(nn.Module):
    def __init__(self,params):
        super(BiencoderRanker,self).__init__()
        self.params=params
        # self.ctxt_bert=BertModel_new.from_pretrained(params['context_bert_model'],config=BertConfig_new.from_pretrained(params['context_bert_model']))
        self.ctxt_bert = load_model('bert-large-uncased',params['context_bert_model'])
        self.cand_encoder=wiki_encoder.WikiEncoderModule(params)
        self.load_cand_encoder_state(params)
        self.context_encoder = BertEncoder(
            self.ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.additional_linear = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.mention_score=MentionScoresHead(bert_output_dim=1024, score_method='qa_mlp', max_mention_length=10)
        self.mention_loss = MentionLoss()

        # self.change_mention_embedding_dim = nn.Linear(768,1024)
        # self.change_mention_embedding_dim=nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 1024),
        # )
        #冻结参数
        for param in self.cand_encoder.parameters():
            param.requires_grad = False


    def load_cand_encoder_state(self,params):
        self.cand_encoder.load_state_dict(torch.load(self.params['cand_encoder']))

    def get_raw_embedding_ctxt(self,input_ids,segment_type,input_mask):
        #返回每个token的嵌入
        raw_ctxt_embedding, _ = self.context_encoder.bert_model(input_ids, segment_type, input_mask)
        #DIM: (bs, seq_len,output_dim)
        return raw_ctxt_embedding

    def get_embedding_mention(self,bert_output,mention_scores,mention_bounds,method='average_all'):
        '''
        mention_bounds=torch.tensor([[[0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[1, 0],[1, 1],[1, 2],[1, 3],[1, 4],
        [2, 0],[2, 1],[2, 2],[2, 3],[2, 4],[3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[4, 0],
        [4, 1],[4, 2],[4, 3],[4, 4]],[[0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[1, 0],[1, 1],[1, 2],[1, 3],[1, 4],
        [2, 0],[2, 1],[2, 2],[2, 3],[2, 4],[3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[4, 0],
        [4, 1],[4, 2],[4, 3],[4, 4]]])
        mention_scores=torch.ones(2,25)
        bert_output=torch.ones(2,5,10)
        '''
        #这里暂时先只实现'average_all'方法
        #先去掉start_pos>end_pos的mention, 以免做重复计算
        mask=(mention_bounds[:,:,1]-mention_bounds[:,:,0])>=0
        mention_bounds=mention_bounds[mask].view(mask.size(0),-1,2)
        mention_scores=mention_scores[mask].view(mask.size(0),-1)

        '''
        example:
            bert_output=torch.ones(1,6,8)
            mention_bounds=torch.tensor([[[0,2],[3,4],[4,2]]])
            
        '''
        if method=='average_all':
            start_pos=mention_bounds[:,:,0]
            end_pos=mention_bounds[:,:,1]
        elif method=='extend':
            start_pos = mention_bounds[:, :, 0]-1
            end_pos = mention_bounds[:, :, 1]+1
            start_pos[start_pos<=0]=1
            end_pos[end_pos>=bert_output.size(1)-1]=bert_output.size(1)-2
        else:
            raise NotImplementedError

        mention_embedding=torch.zeros(mention_bounds.size(0),mention_bounds.size(1),bert_output.size(2),dtype=bert_output.dtype,device=bert_output.device)
        for i in range(mention_bounds.size(0)):
            for j in range(mention_bounds.size(1)):
                cur_star=start_pos[i,j]
                cur_end=end_pos[i, j]
                mention_embedding[i,j,:] = bert_output[i,cur_star:cur_end+1,:].mean(dim=0)


        return mention_embedding,mention_scores,mention_bounds


    def get_embedding_cand(self,input_ids, segment_type, input_mask):
        #返回candidate的整句嵌入
        #DIM: (bs, output_dim)
        return self.cand_encoder(input_ids,segment_type,input_mask)

    def get_mention_scores(self,bert_output,input_mask):

        #return mention_scores, mention_bounds
        return self.mention_score(bert_output,input_mask.bool())

    def forward_ctxt(self,input_ids,segment_type,input_mask,EL=False):

        bert_output_ctxt=self.get_raw_embedding_ctxt(input_ids,segment_type,input_mask)
        mention_scores, mention_bounds=self.get_mention_scores(bert_output_ctxt,input_mask.bool())
        if EL:
            mention_embedding,mention_scores,mention_bounds=self.get_embedding_mention(bert_output_ctxt,mention_scores,mention_bounds,method='extend')
            mention_embedding = self.additional_linear(self.dropout(mention_embedding))
        else:
            mention_embedding=None

        return bert_output_ctxt, mention_scores,mention_bounds,mention_embedding

    def prune_mention(
        self,
        mention_scores,
        mention_bounds,
        mention_embedding,
        gold_mention_bounds,
        gold_mention_bounds_mask,
        gold_entity_local_id
    ):

        '''
        example:
        这里，我只筛掉scores==-float('inf'), end_pos<start_pos的mention，没有进一步根据threshold score筛选，先尽量简单化以跑通逻辑
            mention_embedding=torch.ones(2,7,10)
            mention_scores=torch.tensor([[2,4,0.5,1,-float('inf'),8,-float('inf')],[2,6,0.5,3,7,8,-float('inf')]])
            mention_bounds=torch.tensor([[[1,2],[3,1],[2,3],[2,5],[3,4],[4,1],[4,5]],[[1,2],[4,1],[2,4],[2,5],[3,4],[5,1],[4,5]]])


            gold_mention_bounds=torch.tensor([[[2,4],[-1,-1],[-1,-1]],[[2,5],[-1,-1],[-1,-1]]])
            gold_mention_bounds_mask=torch.tensor([[1,0,0],[1,0,0]]).bool()

            gold_entity_local_id=torch.tensor([[200,-1,-1],[205,-1,-1]])
        '''
        gold_mention_bounds2=torch.clone(gold_mention_bounds).to(gold_mention_bounds.device)
        gold_mention_bounds2[:, :, 1] -= 1
        gold_mention_bounds2[~(gold_mention_bounds_mask.bool())] = -1  # ensure don't select masked to score

        gold_entity_local_id_extend = torch.zeros_like(mention_scores,device=gold_entity_local_id.device)
        for i in range(mention_bounds.size(0)):
            for j in range(mention_bounds.size(1)):
                link=False
                for m in range(gold_mention_bounds2.size(1)):
                    if gold_mention_bounds2[i,m,0]<0:
                        continue
                    if mention_bounds[i,j,0]==gold_mention_bounds2[i,m,0] and mention_bounds[i,j,1]==gold_mention_bounds2[i,m,1]:
                        gold_entity_local_id_extend[i,j]=gold_entity_local_id[i,m]
                        link=True
                if not link:
                    gold_entity_local_id_extend[i, j]=-1


        mask = (mention_scores != -float('inf')) & (mention_bounds[:, :, 1] >= mention_bounds[:, :, 0])
        #DIM:(all_pred_mention_in_batch, 1)
        mention_scores = mention_scores[mask].view(-1,1)
        #DIM:(all_pred_mention_in_batch, 2)
        mention_bounds = mention_bounds[mask]
        #DIM: (all_pred_mention_in_batch, output_dim)
        mention_embedding=mention_embedding[mask]
        #DIM: (all_pred_mention_in_batch, 1)
        gold_entity_local_id_extend=gold_entity_local_id_extend[mask].view(-1,1)
        del gold_mention_bounds2
        return mention_scores, mention_bounds,mention_embedding, gold_entity_local_id_extend

    def forward_cand(
        self,
        input_ids,
        segment_type,
        input_mask,
        pre_computed_cand_embedding=None
    ):
        #pre_computed_cand_embedding；在本代码中为用blink的wiki_encoder预先计算的embedding

        if pre_computed_cand_embedding==None:
                embedding=pre_computed_cand_embedding
        else:
                embedding=self.get_embedding_cand(input_ids, segment_type, input_mask)
        return embedding

    def forward(
        self,
        input_ids_ctxt,

        input_mask_ctxt,
        gold_mention_bounds,
        gold_mention_bounds_mask,
        input_ids_cand=None,
        segment_type_cand=None,
        input_mask_cand=None,
        return_forward_ctxt=True,
        pre_trained_cand=None,
        all_mention_embedding=None,
        candidate_label=None,
        EL=False
        ):

        input_ids_ctxt = torch.LongTensor(input_ids_ctxt).to(device)

        input_mask_ctxt = torch.tensor(input_mask_ctxt).to(device)
        segment_type_ctxt = torch.ones(input_ids_ctxt.size(), dtype=torch.int8).to(device)
        gold_mention_bounds = torch.clone(torch.tensor(gold_mention_bounds)).to(device)
        gold_mention_bounds_mask = torch.tensor(gold_mention_bounds_mask).to(device)


        if return_forward_ctxt:
            bert_output_ctxt, mention_scores,mention_bounds, mention_embedding=self.forward_ctxt(input_ids_ctxt, segment_type_ctxt, input_mask_ctxt,EL)

            # compute Mention Loss

            ment_loss = self.mention_loss(gold_mention_bounds, gold_mention_bounds_mask.bool(), mention_scores, mention_bounds)
            self.ment_loss=ment_loss
            return bert_output_ctxt, mention_scores,mention_bounds, mention_embedding, ment_loss

        if pre_trained_cand is None:
            cand_embedding=self.forward_cand(input_ids_cand, segment_type_cand, input_mask_cand)
        else:
            cand_embedding=pre_trained_cand



        #计算candidate entities的分数
        scores=self.score_cand(all_mention_embedding,cand_embedding)

        #compute EL Loss
        #DIM (all_predmention*top_k,1)
        scores=scores.squeeze(2)
        '''
            candidate_label=torch.tensor([[0,1],[1,0],[0,0]])
        '''
        candidate_label=candidate_label
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
        # print("scores.shape: ",scores.shape)
        # print("label.shape: ",candidate_label.shape)
        el_loss=loss_fct(scores.float(),candidate_label.float())
        return el_loss+self.ment_loss

    def score_cand(self,mention_embedding,cand_embedding):
        '''
        mention_embedding=torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
        cand_embedding=torch.tensor([[[1,1,1,1,1,1],[1,1,1,1,1,1]],[[1,1,1,1,1,1],[1,1,1,1,1,1]],[[1,1,1,1,1,1],[1,1,1,1,1,1]]])
        scores=torch.bmm(cand_embedding, mention_embedding.unsqueeze(2))

        :param mention_embedding:
        :param cand_embedding:
        :return:
        '''
        #DIM (all_predmention, top_k)
        scores = torch.bmm(cand_embedding, mention_embedding.unsqueeze(2))

        return scores



from faiss_indexer import DenseIndexer,DenseFlatIndexer,DenseIVFFlatIndexer,DenseHNSWFlatIndexer
import json
import time
from mention_detection.mention_data_proc_all import ReadTrainDectMent,IterData
from mention_detection.utils import *
import torch.optim as optim
top_k=1
#torch.cuda.set_device(0)
num_epoch = 3
batch_size = 16
eval_n=1
LR=0.005
el=True
device = 'cpu'

def predict(mention_scores, mention_bounds, top_k):
    # DIM:(bs,2), '2' for two indexes of max scores
    top_k_idx = mention_scores.topk(top_k)[1]
    # DIM:(bs,top_k,2)
    top_k_bounds = torch.zeros(mention_bounds.size(0), top_k, 2, dtype=torch.long)
    for i in range(mention_scores.size(0)):
        top_k_bounds[i, :, :] = mention_bounds[i, top_k_idx[i], :]

    return top_k_bounds

def main():
    biencoder_params=json.load(open('./model/biencoder/wiki_encoder_large2.json'))
    ranker=BiencoderRanker(biencoder_params).to(device)
 
    trainfile_path='./Data/train.jsonl'
    train_data=ReadTrainDectMent(trainfile_path,True)
    train_data.padding()


    optimizer=optim.AdamW(ranker.parameters(), lr=LR)
    num_batch=train_data.data_size//batch_size
    if el:
        st = time.time()
        # 加载所有wiki embedding
        all_embedding = torch.load('/public/home/wyop/pqq/all_entities_large.t7')
        end = time.time()
        print("read all entities done. duration: ",end - st)

        st = time.time()
        # 读取index
        indexer = DenseFlatIndexer(1, 50000)
        indexer.deserialize_from('/public/home/wyop/pqq/faiss_hnsw_index.pkl')
        end = time.time()
        print("read indexer done. duration: ", end - st)
    for epoch in range(num_epoch):
        n=1
        dataload = IterData(train_data, batch_size, True, True)
        for input_ids_ctxt,input_mask_ctxt,gold_mention_bounds,gold_mention_bounds_mask,gold_entity_local_id in dataload:
            st_all = time.time()
            gold_entity_local_id = torch.tensor(gold_entity_local_id, device=device)
            gold_mention_bounds2 = torch.clone(torch.tensor(gold_mention_bounds)).to(device)
            optimizer.zero_grad()

            '''
            forward():
                input_ids_ctxt,
                segment_type_ctxt,
                input_mask_ctxt,
                
                gold_mention_bounds,
                gold_mention_bounds_mask,
                
                input_ids_cand=None,
                segment_type_cand=None,
                input_mask_cand=None,
                
                return_forward_ctxt=True,
                pre_trained_cand=None
                
                all_mention_embedding=None,
                candidate_label=None
                
            '''
            # 第一个返回的是bert_output, 后面暂时用不上
            #mention_embedding: DIM(bs, pred_mention,output_dim)
            bert_output, mention_scores,mention_bounds,mention_embedding,ment_loss=ranker(
                                                               input_ids_ctxt, input_mask_ctxt,
                                                               gold_mention_bounds,gold_mention_bounds_mask,
                                                               None,None,None,
                                                               True,None,
                                                               None,None,el
                                                               )
            #print(bert_output.shape)
            top_k_bounds = predict(mention_scores, mention_bounds, top_k)
            pres, rec = compute_precise_recall_overlap2(top_k_bounds, torch.tensor(gold_mention_bounds2))
            if el:

                #print(mention_embedding.shape)
                #DIM: mention_embedding(all_pred_mention_in_batch, output_dim)
                mention_scores, mention_bounds,mention_embedding,gold_entity_local_id_extend=ranker.prune_mention(
                                                                                             mention_scores, mention_bounds, mention_embedding,
                                                                                             gold_mention_bounds2, torch.tensor(gold_mention_bounds_mask), gold_entity_local_id

                                                                                 )
                mask = (gold_entity_local_id_extend != -1).squeeze(1)
                mention_scores=mention_scores[mask]
                mention_bounds = mention_bounds[mask]
                gold_entity_local_id_extend = gold_entity_local_id_extend[mask]
                mention_embedding=mention_embedding[mask]


                '''
                example:
                gold_entity_local_id_extend=torch.tensor([[20],[-1],[4555]])
                mention_bounds=torch.tensor([[2,3],[3,4],[5,5]])
                mention_embedding=torch.rand(3,10)
                mention_scores=torch.tensor([[1],[2],[3]])
                
                mask=(gold_entity_local_id_extend!=-1).squeeze(1)
                gold_entity_local_id_extend=gold_entity_local_id_extend[mask]
                mention_bounds=mention_bounds[mask]
                
                '''

                #print(mention_embedding.shape)

                #为每个mention查找最相近的10个entity
                #mention_embedding=torch.randn(5,1024)
                #gold_entity_local_id_extend=torch.tensor([22531,50,-1,49902,-1]).view(-1,1)
                #DIM: scores(all_predmention_in_batch,10), indexs(indexs,10)
                #type: "numpy.array"
                st = time.time()
                scores,indexs=indexer.search_knn(mention_embedding.cpu().detach().numpy(), 8)
                end = time.time()
                print("pred_mention_num: ", mention_embedding.shape[0]," index time: ",end - st)

                #给indexs中的gold entity附上标签 '1',注意这个操作并不能保证将所有的gold mention的gold entity标出来
                #因为找出的最相近的是个中可能并没有找出gold entity，还需要后续操作。
                indexs=torch.from_numpy(indexs).to(gold_entity_local_id_extend.device)
                #DIM: (all_predmention_in_batch,10)
                index_label=torch.zeros_like(indexs,dtype=torch.int8,device=indexs.device)
                index_label[indexs==gold_entity_local_id_extend]=1

                #先找出没有gold entity的mention，并用gold_entity_local_id_extend中的entity将其替换，并将label赋为'1'
                #这样的操作也会将'-1'的entity 添加进去，后续再做一个操作就可以排除
                #bool tensor, DIM (all_predmention_in_batch,1)
                need_add_gold_pos=(index_label.sum(dim=1)==0).view(-1,1)
                #indexs[need_add_gold_pos.squeeze(1)][:,0]=gold_entity_local_id_extend[need_add_gold_pos]
                for i in range(indexs.size(0)):
                    if not need_add_gold_pos[i]:
                        continue
                    else:
                        indexs[i,7]=gold_entity_local_id_extend[i]
                        index_label[i,7]=1

                #找出 entity为-1的位置，然后赋一个随机的entity，再将标签改为0
                invalid_entity_mask= indexs==-1
                indexs[invalid_entity_mask]=torch.randint(0,5900000,(1,1)).item()
                index_label[invalid_entity_mask]=0
                # print("indexs: ",indexs)
                # print("index_label: ",index_label)
                #获取candidate embedding
                candidate_embedding = all_embedding[indexs.to(mention_embedding.device)].to(mention_embedding.device)

                biloss=ranker(
                       input_ids_ctxt, input_mask_ctxt,
                       gold_mention_bounds2,gold_mention_bounds_mask,
                       None,None,None,
                       False,candidate_embedding,
                       mention_embedding,index_label
                       )
            if not el:
                ment_loss.backward()
            else:
                print("start backward.")
                st = time.time()
                biloss.backward()
                print("end backward.")
                end = time.time()
                print("backward duration: ",end - st)
            optimizer.step()
            if n%eval_n==0:
                print('*'*25,'epoch: ',epoch+1,'/',num_epoch,'\t','batch: ',n,'/',num_batch,'*'*25)
                print('pres: ',pres)
                print('recall: ',rec)
                print('ment_loss: ',ment_loss)
                if el:
                 print('el_loss: ',biloss-ment_loss)
                 print('biloss: ',biloss)
                print('\n')
            n+=1
            end_all = time.time()
            print("batch duration: ",end_all-st_all)
    torch.save(ranker.state_dict(),'./model/mybiencoder_wiki2.bin')
if __name__=='__main__':
    main()
