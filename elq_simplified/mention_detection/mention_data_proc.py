import json
from torch.utils.data import Dataset,DataLoader
import random
class ReadTrainDectMent(Dataset):
    def __init__(self,file_path,destroy_raw_data=True):
        self.file_path=file_path
        self.raw_data=self.read_jsonl()
        self.data_size=len(self.raw_data)
        self.max_tokens_len = None #not +2 for [CLS] and [SEP]
        self.max_num_gold_mention=None
        self.QuesToken_ids, self.GoldMentSpan_idx,self.MentToEntSampleID=self.get_ques_GoldMentSpan()
        #notice that the difference between the suffix ids and idx above.
        self.input_mask=None
        self.GoldMentSpan_idx_mask=None

        self.destroy_RawData(destroy_raw_data)

    def __getitem__(self, item):
        return self.QuesToken_ids[item], self.GoldMentSpan_idx[item]

    def __len__(self):
        return self.data_size

    def read_jsonl(self):
        '''
        :return: list[dict], dict: one entry in jsonl file
        '''
        data = []
        with open(self.file_path,'r',encoding='utf-8') as f:
            for l in f:
                data.append(json.loads(l))
        return data

    def get_ques_GoldMentSpan(self):
        '''
        :return: QuesToken_ids: 深度为2的list
                 GoldMentionSpan_idx: 深度为2的list
        '''
        QuesToken_ids = []
        GoldMentionSpan_idx=[]
        MentToEntSampleID=[]
        max_len=-100
        max_num_gold_mention=-100
        # min_len=float('inf')
        for i in self.raw_data:
            QuesToken_ids.append(i['tokenized_text_ids'])
            GoldMentionSpan_idx.append(i['tokenized_mention_idxs'])
            MentToEntSampleID.append(i['sample_id'])
            if len(QuesToken_ids[-1])>max_len:
                max_len=len(QuesToken_ids[-1])
            if len(i['tokenized_mention_idxs'])>max_num_gold_mention:
                max_num_gold_mention=len(i['tokenized_mention_idxs'])
            # if len(QuesToken_ids[-1])<min_len:
            #     min_len=len(QuesToken_ids[-1])
        self.max_tokens_len=max_len
        self.max_num_gold_mention=max_num_gold_mention
        # self.min_len=min_len
        return QuesToken_ids, GoldMentionSpan_idx,MentToEntSampleID

    def padding(self):
        '''
        1. add '101'([CLS]) and '102'([SEP]) for every token_ids vector,
        and then pad the short token_ids vector with '0',
        meanwhile, generate an input_mask vector mask which token is the padded one,
        '1' for not padding, '0' for padding.

        2. pad the short GoldMentSpan_idx with [-1,-1], and generate a GoldMentSpan_idx_mask tensor
           '1' for not padding, '0' for padding.

        3. pad the MentToEntSampleID with -1.
        '''
        input_mask=[]  #alias: attention_mask
        GoldMentSpan_idx_mask=[]
        for i in range(self.data_size):
            #padding QuesToken_ids and generate input_mask tensor
            len_cur=len(self.QuesToken_ids[i])
            self.QuesToken_ids[i]= [101] + self.QuesToken_ids[i]+[102] + [0]*(self.max_tokens_len-len_cur)
            input_mask_one=[1]*(len_cur+2) + [0]*(self.max_tokens_len-len_cur)
            input_mask.append(input_mask_one)

            # padding GoldMentSpan_idx
            for n in range(len(self.GoldMentSpan_idx[i])):
                for m in range(2):
                    self.GoldMentSpan_idx[i][n][m]+=1 #add 1,since had added 'CLS' to input seq

            num_gold_cur=len(self.GoldMentSpan_idx[i])
            self.GoldMentSpan_idx[i]+=[[-1,-1]]*(self.max_num_gold_mention-num_gold_cur)
            GoldMentSpan_idx_mask.append([1]*num_gold_cur+[0]*(self.max_num_gold_mention-num_gold_cur))

            #padding MentToEntSampleID
            self.MentToEntSampleID[i]+=[-1]*(self.max_num_gold_mention-num_gold_cur)

        self.input_mask=input_mask
        self.GoldMentSpan_idx_mask=GoldMentSpan_idx_mask

    def destroy_RawData(self,tag):
        if tag:
            del self.raw_data


class IterData():
    def __init__(self,data:ReadTrainDectMent,batch_size=50,shuffle=False,drop=False):
        self.data=data
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.drop=drop
        self.cur_pos=0
        self.data_size=data.data_size

    def __iter__(self):
        if not self.shuffle:
            while self.cur_pos<self.data_size:
                if self.drop:
                    if self.data_size-self.cur_pos<self.batch_size:
                        break
                one_batch=(self.data.QuesToken_ids[self.cur_pos:self.cur_pos+self.batch_size],
                           self.data.input_mask[self.cur_pos:self.cur_pos+self.batch_size],
                           self.data.GoldMentSpan_idx[self.cur_pos:self.cur_pos+self.batch_size],
                           self.data.GoldMentSpan_idx_mask[self.cur_pos:self.cur_pos+self.batch_size],
                           self.data.MentToEntSampleID[self.cur_pos:self.cur_pos+self.batch_size]
                           )
                self.cur_pos+=self.batch_size
                yield one_batch
        else :
            rand_idx=random.sample(range(0,self.data_size),self.data_size)
            while self.cur_pos<self.data_size:
                if self.drop:
                    if self.data_size-self.cur_pos<self.batch_size:
                        break
                ques_token_ids=[]
                input_mask = []
                gold_mention_idx=[]
                gold_mention_idx_mask=[]
                ment_to_entity=[]
                for i in range(self.batch_size):
                    if i+self.cur_pos >= self.data_size:
                        break
                    ques_token_ids.append(self.data.QuesToken_ids[rand_idx[self.cur_pos+i]])
                    input_mask.append(self.data.input_mask[rand_idx[self.cur_pos+i]])
                    gold_mention_idx.append(self.data.GoldMentSpan_idx[rand_idx[self.cur_pos+i]])
                    gold_mention_idx_mask.append(self.data.GoldMentSpan_idx_mask[rand_idx[self.cur_pos+i]])
                    ment_to_entity.append(self.data.MentToEntSampleID[rand_idx[self.cur_pos+i]])
                self.cur_pos+=self.batch_size
                yield (ques_token_ids,input_mask,gold_mention_idx,gold_mention_idx_mask,ment_to_entity)


# train_file_path = '../Data/train_merge.jsonl'
# traindata=ReadTrainDectMent(train_file_path)
# traindata.padding()
# dataloader=IterData(traindata)
# n=0
# for i in dataloader:
#     if n==0:
#         print(i[4])
#     n+=1


