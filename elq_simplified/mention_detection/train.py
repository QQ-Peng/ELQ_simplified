from utils import *
from mention_detection import *
import torch.optim as optim
import time
from mention_data_proc_all import *

def train_model(bert_model_name,
                bert_model_path,
                train_file_path,
                batch_size,
                learning_rate,
                top_k,
                device,
                bert_dim,
                n_epoch,
                score_method,
                mention_maxlen=10
                ):
    traindata = ReadTrainDectMent(train_file_path)
    traindata.padding()
    mend_dect_model=MentionDect(bert_model_name,bert_model_path,bert_dim,score_method,mention_maxlen).to(device)
    optimizer = optim.SGD(mend_dect_model.parameters(),lr=learning_rate)
    loss_step = []
    loss_all = []
    top_k_bounds = None
    bounds = None
    gold_mention_bounds = None
    precise = []
    recall = []
    for epoch in range(n_epoch):
        dataloader = IterData(traindata,batch_size, True, True)
        loss_one = []
        n=1
        for input_ids,input_mask,gold_mention_idx,gold_mention_idx_mask,_ in dataloader:
            start = time.time()
            optimizer.zero_grad()
            loss,mention_scores,mention_bounds = mend_dect_model(input_ids,input_mask,gold_mention_idx,gold_mention_idx_mask,device)
            top_k_bounds = mend_dect_model.predict(mention_scores,mention_bounds,top_k)

            loss_one.append(loss)
            loss_all.append(loss)
            loss.backward()
            optimizer.step()
            end = time.time()

            gold_mention_bounds=torch.tensor(gold_mention_idx)
            if n % 4 == 0:
                print("epoch: ", epoch + 1, '/', n_epoch, '\t', "batch: ", n, '/', traindata.data_size // batch_size)
                print('\t', "loss: ", loss.item(), '\t', 'time: cost: ', end - start)
                pres,rec,_=compute_precise_recall_overlap(top_k_bounds,gold_mention_bounds)
                precise.append(pres)
                recall.append(rec)
                print('\t',"precise: ", pres, '\t', 'recall: ', rec)
                if top_k > 1:
                    print('Warning: top_k>1! 未完全剔除不合理的bounds，recall值仅供参考！')
                print('\n')
            n += 1

        loss_step.append(loss_one)
    return mend_dect_model