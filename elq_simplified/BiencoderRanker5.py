# Date: 2020/11/28
# Author: Qianqian Peng

##################################################version-5#############################################################
#在第四版中，固定了wiki encoder的参数，这一版，准备也更新wiki encoder的参数。
#最后发现是因为对loss函数理解不深导致出错。（睡前静心思考的结果）
#
#第五版定为初步最终版。
#
'''
教训：
1. 慎用pycharm自动补全功能。因为在第一版中，pycharm给我把变量input_ids补全成了input_mask而我没有看出来，
   找这个错找了一天半，最后发现是这样的错误，差点被自己蠢哭。
2. 弄清base与large分词的区别，最后发现没有区别，弄清新旧版transformer读取bert模型后对同一输入embedding的区别，
   最后发现没有区别，唯一的区别就是float型精度带来的影响。
3. 弄清源代码调用的bert模型。
4. batch size不能大，虽然最终得到的嵌入可能占不了多大内存，但是BERT是一个并行化的模型，batch size大了之后，涉及到的参数量很大
  （这是我的经验心得，具体得看bert源代码）。
5. 用torch编程，尽量用矩阵运算，规避自己写循环，即使这样会做很多无用操作，但是在用GPU加速的情况下，这种浪费是值得的。
   即便如此，也得优化代码，尽量减少无用计算。
6. 很重要的一条是，深刻理解loss函数。
7. 最重要的一条是，不要太陷入代码，得抽出身来静心思考。
'''
##################################################version-5#############################################################



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

def gener_segmt_mask(token_ids):
    input_mask=torch.ones(token_ids.shape,dtype=torch.int64,device=token_ids.device)
    input_mask[token_ids==0]=0
    segment_ids=torch.ones(token_ids.shape,dtype=torch.int64,device=token_ids.device)
    return segment_ids, input_mask

def load_id2Tokenids(token_ids):
    id2tokenids={}
    for i in range(len(token_ids.size(0))):
        id2tokenids[i]=token_ids[i,:].tolist()
    return id2tokenids

class BiencoderRanker(nn.Module):
    def __init__(self,params):
        super(BiencoderRanker,self).__init__()
        self.params=params
        # self.ctxt_bert=BertModel_new.from_pretrained(params['context_bert_model'],config=BertConfig_new.from_pretrained(params['context_bert_model']))
        self.ctxt_bert = load_model('./model/bert-large-uncased',params['context_bert_model'])
        self.cand_encoder=wiki_encoder.WikiEncoderModule(params)
        # self.load_cand_encoder_state()
        self.context_encoder = BertEncoder(
            self.ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.mention_score=MentionScoresHead(bert_output_dim=768, score_method='qa_linear', max_mention_length=10)
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
            param.requires_grad = True


    def load_cand_encoder_state(self):

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

    def forward_ctxt(self,input_ids,segment_type,input_mask,EL=False,mention_embedding_method='extend'):

        bert_output_ctxt=self.get_raw_embedding_ctxt(input_ids,segment_type,input_mask)
        mention_scores, mention_bounds=self.get_mention_scores(bert_output_ctxt,input_mask.bool())
        if EL:
            mention_embedding,mention_scores,mention_bounds=self.get_embedding_mention(bert_output_ctxt,mention_scores,mention_bounds,mention_embedding_method)
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
        EL=False,
        mention_embedding_method='extend'
        ):

        input_ids_ctxt = torch.LongTensor(input_ids_ctxt).to(device)

        input_mask_ctxt = torch.tensor(input_mask_ctxt).to(device)
        segment_type_ctxt = torch.ones(input_ids_ctxt.size(), dtype=torch.int8).to(device)
        gold_mention_bounds = torch.clone(torch.tensor(gold_mention_bounds)).to(device)
        gold_mention_bounds_mask = torch.tensor(gold_mention_bounds_mask).to(device)


        if return_forward_ctxt:
            bert_output_ctxt, mention_scores,mention_bounds, mention_embedding=self.forward_ctxt(input_ids_ctxt, segment_type_ctxt, input_mask_ctxt,EL,mention_embedding_method)

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
        #DIM (all_predmention,top_k)
        scores=scores.squeeze(2)
        '''
            candidate_label=torch.tensor([[0,1],[1,0],[0,0]])
        '''
        candidate_label=candidate_label
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
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
from mention_detection.mention_data_proc import ReadTrainDectMent,IterData
from mention_detection.utils import *
import torch.optim as optim
top_k=1
torch.cuda.set_device(1)
num_epoch=8
batch_size=8
eval_n=4
LR=0.01
el=True
rand_negative=5
device='cuda'

def predict(mention_scores, mention_bounds, top_k):
    # DIM:(bs,2), '2' for two indexes of max scores
    top_k_idx = mention_scores.topk(top_k)[1]
    # DIM:(bs,top_k,2)
    top_k_bounds = torch.zeros(mention_bounds.size(0), top_k, 2, dtype=torch.long)
    for i in range(mention_scores.size(0)):
        top_k_bounds[i, :, :] = mention_bounds[i, top_k_idx[i], :]

    return top_k_bounds

def main():
    biencoder_params=json.load(open('./model/biencoder/wiki_encoder_large.json'))
    ranker=BiencoderRanker(biencoder_params).to(device)
 
    trainfile_path='./Data/train_merge.jsonl'
    train_data=ReadTrainDectMent(trainfile_path,True)
    train_data.padding()


    optimizer=optim.SGD(ranker.parameters(), lr=LR)
    num_batch=train_data.data_size//batch_size
    for epoch in range(num_epoch):
        n=1
        dataload = IterData(train_data, batch_size, True, True)
        for input_ids_ctxt,input_mask_ctxt,gold_mention_bounds,gold_mention_bounds_mask,gold_entity_local_id in dataload:

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
                                                               None,None,el,
                                                               mention_embedding_method='extend'
                                                               )
            #print(bert_output.shape)
            top_k_bounds = predict(mention_scores, mention_bounds, top_k)
            pres, rec = compute_precise_recall_overlap2(top_k_bounds, torch.tensor(gold_mention_bounds2))
            if el:




                #在这一版中，每一个batch都会重新计算candidate entity的embedding，所以不需要预先的索引
                # # 读取index
                # indexer = DenseFlatIndexer(1, 5000)
                # indexer.deserialize_from('./Data/index_merge.pkl')

                #print(mention_embedding.shape)
                #DIM: mention_embedding(all_pred_mention_in_batch, output_dim)
                # print('here')
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

                #scores,indexs=indexer.search_knn(mention_embedding.cpu().detach().numpy(), 10)

                #计算所有mention与所有entity的内积
                # all_scores=mention_embedding@all_wiki_embedding.t()
                # _, top_k_indices=all_scores.topk(10)

                #随机取负样本
                indices=torch.randint(0,5475,(mention_embedding.size(0),rand_negative)).to(device)


                #给indexs中的gold entity附上标签 '1',注意这个操作并不能保证将所有的gold mention的gold entity标出来
                #因为找出的最相近的是个中可能并没有找出gold entity，还需要后续操作。
                #DIM: (all_predmention_in_batch,10)
                index_label=torch.zeros_like(indices,dtype=torch.int8,device=indices.device)
                index_label[indices==gold_entity_local_id_extend]=1

                #先找出没有gold entity的gold mention，并用gold_entity_local_id_extend中的entity将其替换，并将label赋为'1'
                #这样的操作也会将'-1'的entity 添加进去，后续再做一个操作就可以排除
                #bool tensor, DIM (all_predmention_in_batch,1)
                need_add_gold_pos=(index_label.sum(dim=1)==0).view(-1,1)
                #indexs[need_add_gold_pos.squeeze(1)][:,0]=gold_entity_local_id_extend[need_add_gold_pos]
                for i in range(indices.size(0)):
                    if not need_add_gold_pos[i]:
                        continue
                    else:
                        indices[i,0]=gold_entity_local_id_extend[i]
                        index_label[i,0]=1

                #找出 entity为-1的位置，然后赋一个随机的entity，再将标签改为0
                invalid_entity_mask= indices==-1
                indices[invalid_entity_mask]=torch.randint(0,5475,(1,1)).item()
                index_label[invalid_entity_mask]=0


                # 计算所有的wiki embedding
                all_token_ids = torch.load('./Data/token_ids_merge_0.0005.t7')
                cand_ids=all_token_ids[indices].view(-1,all_token_ids.size(1)).to(device)
                print("cand_ids.shape: ",cand_ids.shape)
                segment_ids_cand, input_mask_cand = gener_segmt_mask(cand_ids)
                segment_ids_cand=segment_ids_cand.view(-1,all_token_ids.size(1))
                input_mask_cand=input_mask_cand.view(-1,all_token_ids.size(1))
                candidate_embedding = ranker.cand_encoder(cand_ids, segment_ids_cand, input_mask_cand)
                candidate_embedding=candidate_embedding.view(mention_embedding.size(0),rand_negative,-1)
                #获取candidate embedding
                # candidate_embedding=all_wiki_embedding[indices.to(mention_embedding.device)].to(mention_embedding.device)

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
                biloss.backward()
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
    torch.save(ranker.state_dict(),'./model/mybiencoder_wiki.bin')
if __name__=='__main__':
    main()
