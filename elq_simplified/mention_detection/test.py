from utils import *
from mention_data_proc_all import *
from mention_detection import *


def test_model(model,bert_model_name,test_file_path,batch_size,top_k,device):
    testdata = ReadTrainDectMent(test_file_path)
    testdata.padding()
    precise = []
    recall = []
    dataloader2 = IterData(testdata, batch_size, True, True)
    n = 1
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    with torch.no_grad():
        for input_ids, input_mask, gold_mention_idx, gold_mention_idx_mask,_ in dataloader2:
            _, mention_scores, mention_bounds = model(input_ids, input_mask, gold_mention_idx, gold_mention_idx_mask,device)
            gold_mention_bounds = torch.tensor(gold_mention_idx)
            top_k_bounds = model.predict(mention_scores, mention_bounds, top_k)
            pres, rec,removeoverlap= compute_precise_recall_overlap(top_k_bounds, gold_mention_bounds)
            precise.append(pres)
            recall.append(rec)
            print('*' * 20,'batch: ',n,'/',testdata.data_size//batch_size,'*'*20)
            print("precise: ", pres)
            print('recall: ', rec)
            if top_k>1:
                print('Warning: top_k>1! 未完全剔除不合理的bounds，recall值仅供参考！')
            print("a predict example: ",'\n',tokenizer.convert_ids_to_tokens(torch.tensor(input_ids[0])))
            print(input_ids[0])
            print("mention bounds: ", '\n', removeoverlap[0])

            n+=1
            # print(top_k_bounds[0, :, :],'\n')