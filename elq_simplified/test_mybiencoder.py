# Date: 2020/12/2
# Author: Qianqian Peng

def gener_segmt_mask(token_ids):
    input_mask=torch.ones(token_ids.shape,dtype=torch.int64,device=token_ids.device)
    input_mask[token_ids==0]=0
    segment_ids=torch.ones(token_ids.shape,dtype=torch.int64,device=token_ids.device)
    return segment_ids, input_mask

def predict(mention_scores, mention_bounds, top_k):
    # DIM:(bs,top_k), top_k for indexes of the top_k max scores
    top_k_idx = mention_scores.topk(top_k)[1]
    # DIM:(bs,top_k,2)
    top_k_bounds = torch.zeros(mention_bounds.size(0), top_k, 2, dtype=torch.long)
    for i in range(mention_scores.size(0)):
        top_k_bounds[i, :, :] = mention_bounds[i, top_k_idx[i], :]

    return top_k_bounds,top_k_idx

def compute_EL_precision(ranker,indexer,mention_embedding,all_cand_embedding,top_k_bounds,top_k_idx,gold_mention_bounds,mention_to_entity):

    #首先将预测的top_k bounds中的预测正确的mention bounds（即与gold mention bounds有overlap）挑出来，然后获得挑出其embedding、对应的gold entity的ID（这里为sample id）
    #然后用获得的mention embedding与wiki KB（这里为sample wiki KB）进行搜索，找到最相似的一个即为预测的linking结果
    """
    example:
    top_k_bounds=torch.tensor([[[2,4],[6,6]]])
    gold_mention_bounds=torch.tensor([[[2,3],[-1,-1],[-1,-1]]])
    mention_to_entity=torch.tensor([[300,-1,-1]])
    """
    pred_correct_bounds_embedding=[]
    pred_correct_bounds_entity = []

    for i in range(top_k_bounds.size(0)):
        for j in range(top_k_bounds.size(1)):
            for m in range(gold_mention_bounds.size(1)):
                if gold_mention_bounds[i,m,0] <= 0:
                    continue  # 跳过padding的gold mention bounds
                if top_k_bounds[i, j, 0] < gold_mention_bounds[i, m, 0] and top_k_bounds[i, j, 1] < \
                        gold_mention_bounds[i, m, 0]:
                    continue
                elif top_k_bounds[i, j, 0] > gold_mention_bounds[i, m, 1] and top_k_bounds[i, j, 1] > \
                        gold_mention_bounds[i, m, 1]:
                    continue
                else:
                    pred_correct_bounds_embedding.append(mention_embedding[i,top_k_idx[i,j],:].tolist())
                    pred_correct_bounds_entity.append(mention_to_entity[i, m].item())
                    break  # 防止一个mention被linking不止一次


    pred_correct_bounds_embedding=torch.tensor(pred_correct_bounds_embedding).to(device)
    pred_correct_bounds_entity=torch.tensor(pred_correct_bounds_entity).view(1,-1).to(device)
    scores = pred_correct_bounds_embedding @ all_embedding.t()

    _,indices=scores.topk(10)
    num=(indices==pred_correct_bounds_entity.view(-1,1)).float().sum()
    print(num,'\t',pred_correct_bounds_entity.size(1))
    el_pres=num/pred_correct_bounds_entity.size(1)
    return el_pres.item()





import json
import torch
import time
from BiencoderRanker4 import BiencoderRanker
from mention_detection.mention_data_proc_all import ReadTrainDectMent,IterData
from mention_detection.utils import *
from transformers import BertTokenizer
from faiss_indexer import DenseFlatIndexer
#torch.cuda.set_device(1)
tokenizer=BertTokenizer.from_pretrained('./model/bert-large-uncased')

biencoder_params=json.load(open('./model/biencoder/wiki_encoder_large2.json'))
with torch.no_grad():
    ranker=BiencoderRanker(biencoder_params)
    ranker.load_state_dict(torch.load('./model/mybiencoder_wiki.bin'))
    for params in ranker.parameters():
        params.requires_grad=False
    ranker=ranker.to('cpu')

trainfile_path='./Data/train.jsonl'
train_data=ReadTrainDectMent(trainfile_path,True)
train_data.padding()
batch_size=32
dataload = IterData(train_data, batch_size, True, True)
top_k=1
EL=True
device = 'cpu'


# 读取index
indexer = DenseFlatIndexer(1, 50000)
st = time.time()
print("start read indexer")
indexer.deserialize_from('/public/home/wyop/pqq/faiss_hnsw_index.pkl')
end = time.time()
print("read indexer done")
print("duration: ",end-st)
st = time.time()
print("start read all embedding")
all_embedding = torch.load('/public/home/wyop/pqq/all_entities_large.t7')
end = time.time()
print("read all embedding done")
print("duration: ",end-st)
# with torch.no_grad():
#     all_token_ids=torch.load('./Data/token_ids_merge_0.0005.t7').to(device)
#     print('all_token_ids.shape: ',all_token_ids.shape)
#     segment_ids_cand, input_mask_cand = gener_segmt_mask(all_token_ids)
#     all_embedding=[]
#     n=0
#     batch_size=8
#     while n+batch_size<=all_token_ids.size(0):
#         embedding_onebatch=ranker.cand_encoder(all_token_ids[n:n+batch_size,:],
#                                                segment_ids_cand[n:n+batch_size,:],
#                                                input_mask_cand[n:n+batch_size,:])
#         all_embedding.append(embedding_onebatch)
#         n+=batch_size
#
#     while n<all_token_ids.size(0):
#         embedding_one=ranker.cand_encoder(all_token_ids[n,:].unsqueeze(0),
#                                           segment_ids_cand[n,:].unsqueeze(0),
#                                           input_mask_cand[n,:].unsqueeze(0))
#         all_embedding.append(embedding_one)
#         n+=1
#
#     all_embedding=torch.cat(all_embedding).to(device)

print('compute all_embedding done')
print('all_embedding.shape',all_embedding.shape)
with torch.no_grad():
    n=1
    for input_ids_ctxt, input_mask_ctxt, gold_mention_bounds, gold_mention_bounds_mask, gold_entity_local_id in dataload:
        gold_mention_bounds2=torch.tensor(gold_mention_bounds)
        gold_entity_local_id = torch.tensor(gold_entity_local_id, device=device)
        _, mention_scores, mention_bounds, mention_embedding, ment_loss = ranker(
            input_ids_ctxt, input_mask_ctxt,
            gold_mention_bounds, gold_mention_bounds_mask,
            None, None, None,
            True, None,
            None, None,EL
        )


        top_k_bounds,top_k_idx = predict(mention_scores, mention_bounds, top_k)
        pres, rec, removeoverlap = compute_precise_recall_overlap(top_k_bounds, gold_mention_bounds2)
        print('*' * 20, 'batch: ', n, '/', train_data.data_size // batch_size, '*' * 20)
        el_pres=compute_EL_precision(ranker,indexer,mention_embedding,all_embedding,top_k_bounds,top_k_idx,gold_mention_bounds2,gold_entity_local_id)
        print("precise: ", pres)
        print('recall: ', rec)
        print('EL pres: ',el_pres)
        if top_k > 1:
            print('Warning: top_k>1! 未完全剔除不合理的bounds，recall值仅供参考！')
        print("a predict example: ", '\n', tokenizer.convert_ids_to_tokens(torch.tensor(input_ids_ctxt[0])))
        print(input_ids_ctxt[0])
        print("mention bounds: ", '\n', removeoverlap[0])
        n+=1
