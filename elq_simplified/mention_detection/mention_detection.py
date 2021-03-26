import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch.nn as nn
import matplotlib.pyplot as plt

def load_model(model_name, model_path):
    model_config = BertConfig.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_path, config=model_config)
    return bert_model


class MentionScoresHead(nn.Module):
    def __init__(self, bert_output_dim=768, score_method='qa_linear', max_mention_length=10):
        super(MentionScoresHead, self).__init__()
        self.max_mention_length = max_mention_length
        self.score_method = score_method
        if self.score_method == 'qa_linear':
            self.score_one_token = nn.Linear(bert_output_dim, 3)
        elif self.score_method == 'qa_mlp' or self.score_method == 'qa':
            self.score_one_token = nn.Sequential(
                nn.Linear(bert_output_dim, bert_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bert_output_dim, 3),
            )
        else:
            raise NotImplementedError()

    def forward(self, bert_output, input_mask):
        '''
                Retuns scores for *inclusive* mention boundaries
        '''
        # (bs, seqlen, 3)
        logits = self.score_one_token(bert_output)
        # (bs, seqlen, 1); (bs, seqlen, 1); (bs, seqlen, 1)
        start_logprobs, end_logprobs, mention_logprobs = logits.split(1, dim=-1)
        # (bs, seqlen)
        start_logprobs = start_logprobs.squeeze(-1)
        end_logprobs = end_logprobs.squeeze(-1)
        mention_logprobs = mention_logprobs.squeeze(-1)
        # impossible to choose masked tokens as starts/ends of spans
        start_logprobs[~input_mask] = -float("Inf")
        end_logprobs[~input_mask] = -float("Inf")
        mention_logprobs[~input_mask] = -float("Inf")
        '''
            经过我的测试，这里的input_mask应该为bool张量 DIM：(bs, seqlen)
            example:
            start_logprobs=torch.tensor([[-0.3832,  0.3092, -0.0431, -0.1295,  0.7214]])
            input_mask=torch.tensor([[True,True,True,True,False]])
            #~input_mask=tensor([[False, False, False, False,  True]])
            start_logprobs[~input_mask]=-float("Inf")
            print(start_logprobs)
            tensor([[-0.3832,  0.3092, -0.0431, -0.1295, -inf]])

            如果input_mask=torch.tensor([[1,1,1,1,0]])
            #则~input_mask = tensor([[-2, -2, -2, -2, -1]])#这里我又有个疑惑，编程语言对有符号整数的取反(~)与数字逻辑教材的算法有差别，正数的原码反码补码都是一样的
            #而这里不是
        '''
        # take sum of log softmaxes:
        # log p(mention) = log p(start_pos && end_pos) = log p(start_pos) + log p(end_pos)
        # DIM: (bs, starts, ends)
        mention_scores = start_logprobs.unsqueeze(2) + end_logprobs.unsqueeze(1)  # end + start
        '''
        example:
        start_logprobs=torch.tensor([[1,2,3,4,5.0]]) #DIM: (1,5)
        end_logprobs=torch.tensor([[6,7,8,9,10.0]]) #DIM: (1,5)
        mention_logprobs=torch.tensor([[0,1,2,3,4.0]])  #DIM: (1,5)
        #start_logprobs.unsqueeze(2)=tensor([[[1],[2],[3],[4],[5]]]) #DIM: (1,5,1)
        #end_logprobs.unsqueeze(1)=tensor([[[ 6,  7,  8,  9, 10]]]) #DIM: (1,1,5)
        mention_scores=start_logprobs.unsqueeze(2)+end_logprobs.unsqueeze(1)# DIM: (1,5,5)
        tensor([[[ 7,  8,  9, 10, 11],
                [ 8,  9, 10, 11, 12],
                [ 9, 10, 11, 12, 13],
                [10, 11, 12, 13, 14],
                [11, 12, 13, 14, 15]]])
        '''
        # (bs, starts, ends)
        mention_cum_scores = torch.zeros(mention_scores.size(), dtype=mention_scores.dtype,device=mention_scores.device)
        # add ends
        mention_logprobs_end_cumsum = torch.zeros(input_mask.size(0), dtype=mention_scores.dtype,device=mention_scores.device)
        for i in range(input_mask.size(1)):
            mention_logprobs_end_cumsum += mention_logprobs[:, i]
            mention_cum_scores[:, :, i] += mention_logprobs_end_cumsum.unsqueeze(-1)
        # print(mention_logprobs_end_cumsum.unsqueeze(-1))
        # print(mention_cum_scores)
        # 将所有以第i个token结尾的区间都加上了累计分数，但是没有区分以第j个token为开始的条件，所以后面要减掉一些分数
        # 这样写代码的方式，批量操作，方便、快捷，但是会导致做了很多无用计算，或许这也是源代码需要将数据加入GPU加速的原因吧

        # subtract starts
        mention_logprobs_start_cumsum = torch.zeros(input_mask.size(0), dtype=mention_scores.dtype,device=mention_scores.device)
        for i in range(input_mask.size(1) - 1):
            mention_logprobs_start_cumsum += mention_logprobs[:, i]
            mention_cum_scores[:, (i + 1), :] -= mention_logprobs_start_cumsum.unsqueeze(-1)

        # DIM: (bs, starts, ends)
        mention_scores += mention_cum_scores

        # DIM: (starts, ends, 2) -- tuples of [start_idx, end_idx]
        mention_bounds = torch.stack([
            torch.arange(mention_scores.size(1)).unsqueeze(-1).expand(mention_scores.size(1), mention_scores.size(2)),
            # start idxs
            #
            #             torch.arange(5): tensor([0, 1, 2, 3, 4]) DIM: (starts)
            #             torch.arange(5).unsqueeze(-1):  tensor([[0],[1],[2],[3],[4]]) DIM: (starts,1)
            #             torch.arange(5).unsqueeze(-1).expand(starts,ends):
            #             tensor([[0, 0, 0, 0, 0],
            #                     [1, 1, 1, 1, 1],
            #                     [2, 2, 2, 2, 2],
            #                     [3, 3, 3, 3, 3],
            #                     [4, 4, 4, 4, 4]]) DIM: (starts,ends)
            #
            torch.arange(mention_scores.size(1)).unsqueeze(0).expand(mention_scores.size(1), mention_scores.size(2)),
            # end idxs
        ], dim=-1).to(input_mask.device)
        # torch.stack([a,b],dim=-1)
        # tensor([[[0, 0],[0, 1],[0, 2],[0, 3],[0, 4]],[[1, 0],[1, 1],[1, 2],[1, 3],[1, 4]],[[2, 0],[2, 1],[2, 2],[2, 3],[2, 4]],
        # [[3, 0],[3, 1],[3, 2],[3, 3],[3, 4]],[[4, 0],[4, 1],[4, 2],[4, 3],[4, 4]]]) DIM: (starts,ends,2)

        # DIM: (starts, ends)
        mention_sizes = mention_bounds[:, :, 1] - mention_bounds[:, :, 0] + 1  # (+1 as ends are inclusive)
        # tensor([[1, 2, 3, 4, 5],
        #         [0, 1, 2, 3, 4],
        #         [-1, 0, 1, 2, 3],
        #         [-2, -1, 0, 1, 2],
        #         [-3, -2, -1, 0, 1]])

        # Remove invalids (startpos > endpos, endpos > seqlen) and renormalize

        # DIM: (bs, starts, ends): 为bool常量
        valid_mask = (mention_sizes.unsqueeze(0) > 0) & input_mask.unsqueeze(1)
        # 上面对mention_sizes有一个unsqueeze(0),多了一个bs维，mention_sizes.unsqueeze(0)：DIM: (1,starts,ends)
        # 上面对input_mask有一个unqueeze(1), input_mask.unsqueeze(1): DIM: (bs,1,token_len=starts=ends)
        # DIM: (bs, starts, ends)
        mention_scores[~valid_mask] = -float("inf")  # invalids have logprob=-inf (p=0)
        # tensor([[[7., 9., 12., 16., -inf],
        #          [-inf, 10., 13., 17., -inf],
        #          [-inf, -inf, 13., 17., -inf],
        #          [-inf, -inf, -inf, 16., -inf],
        #          [-inf, -inf, -inf, -inf, -inf]]])

        # DIM: (bs, starts * ends)
        mention_scores = mention_scores.view(mention_scores.size(0), -1)
        # tensor([[7., 9., 12., 16., -inf, -inf, 10., 13., 17., -inf, -inf, -inf, 13., 17.,
        #          -inf, -inf, -inf, -inf, 16., -inf, -inf, -inf, -inf, -inf, -inf]])

        # DIM: (bs, starts * ends, 2)
        mention_bounds = mention_bounds.view(-1, 2)  # DIM: (starts*ends,2)
        # tensor([[0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[1, 0],[1, 1],[1, 2],[1, 3],[1, 4],
        # [2, 0],[2, 1],[2, 2],[2, 3],[2, 4],[3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[4, 0],
        # [4, 1],[4, 2],[4, 3],[4, 4]])
        # DIM: (bs, starts * ends, 2)
        mention_bounds = mention_bounds.unsqueeze(0).expand(mention_scores.size(0), mention_scores.size(1), 2)

        if self.max_mention_length is not None:
            mention_scores, mention_bounds = self.filter_by_mention_size(
                mention_scores, mention_bounds, self.max_mention_length,
            )
        return mention_scores, mention_bounds

    def filter_by_mention_size(self, mention_scores, mention_bounds, max_mention_length):
        '''
        Filter all mentions > maximum mention length
        mention_scores: torch.FloatTensor (bs, num_mentions)
        mention_bounds: torch.LongTensor (bs, num_mentions, 2)
        '''
        # (bs, num_mentions)
        mention_bounds_mask = (mention_bounds[:, :, 1] - mention_bounds[:, :, 0]+1 <= max_mention_length)
        """
        mention_bounds[:,:,1] - mention_bounds[:,:,0]+1:
            tensor([[1, 2, 3, 4, 5, 0, 1, 2, 3, 4, -1, 0, 1, 2, 3, -2, -1, 0,
                        1, 2, -3, -2, -1, 0, 1]])
        (mention_bounds[:,:,1] - mention_bounds[:,:,0]+1 <= 3):
            tensor([[ True,  True,  True, False, False,  True,  True,  True,  True, False,
                        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                        True,  True,  True,  True,  True]])
        """
        # (bs, num_filtered_mentions)
        mention_scores = mention_scores[mention_bounds_mask]
        mention_scores = mention_scores.view(mention_bounds_mask.size(0), -1)  # '-1'代表维度自动推断
        # 这里的例子 维度为(1,22)
        # (bs, num_filtered_mentions, 2)
        mention_bounds = mention_bounds[mention_bounds_mask]
        mention_bounds = mention_bounds.view(mention_bounds_mask.size(0), -1, 2)
        return mention_scores, mention_bounds

class MentionLoss(nn.Module):
    def __int__(self):
            super(MentionLoss, self).__init__()

    def forward(self,gold_mention_bounds, gold_mention_bounds_mask, mention_logits, mention_bounds,):

        return self.get_span_loss(gold_mention_bounds,gold_mention_bounds_mask,mention_logits,mention_bounds)

    def get_span_loss(
            self, gold_mention_bounds, gold_mention_bounds_mask, mention_logits, mention_bounds,
    ):
        """
        gold_mention_bounds (bs, num_mentions, 2)
        gold_mention_bounds_mask (bs, num_mentions):
        mention_logits (bs, all_mentions)
        menion_bounds (bs, all_mentions, 2)
        """
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
        gold_mention_bounds[:,:,1] -= 1
        gold_mention_bounds[~gold_mention_bounds_mask] = -1  # ensure don't select masked to score
        # triples of [ex in batch, mention_idx in gold_mention_bounds, idx in mention_bounds]
        # use 1st, 2nd to index into gold_mention_bounds, 1st, 3rd to index into mention_bounds
        '''example: 
            tokens=tokenizer.tokenize('i love china!')
            gold_mention_bounds=torch.tensor([[[2,2],[3,3]]])
            gold_mention_bounds_mask=torch.tensor([[1,1]]).bool()
            gold_mention_pos_idx = ((mention_bounds.unsqueeze(1) - gold_mention_bounds.unsqueeze(2)).abs().sum(-1) == 0).nonzero():
            [[0,1,12],
             [0,1,18]]             
        '''
        gold_mention_pos_idx = ((
                mention_bounds.unsqueeze(1) - gold_mention_bounds.unsqueeze(2)
                # (bs, num_mentions, start_pos * end_pos, 2)
                ).abs().sum(-1) == 0).nonzero()

        # (bs, total_possible_spans)
        gold_mention_binary = torch.zeros(mention_logits.size(), dtype=mention_logits.dtype).to(gold_mention_bounds.device)
        gold_mention_binary[gold_mention_pos_idx[:, 0], gold_mention_pos_idx[:, 2]] = 1

        # prune masked spans
        mask = mention_logits != -float("inf")
        masked_mention_logits = mention_logits[mask]#.to('cpu')
        masked_gold_mention_binary = gold_mention_binary[mask]#.to('cpu')

        # (bs, total_possible_spans)
        span_loss = loss_fct(masked_mention_logits, masked_gold_mention_binary)

        return span_loss


class MentionDect(nn.Module):
    def __init__(self,base_model_name,base_model_path,bert_output_dim=768,score_method='qa_linear',max_mention_length=10):

        super(MentionDect, self).__init__()
        self.bert_output_dim=bert_output_dim
        self.score_method=score_method
        self.max_mention_length=max_mention_length

        self.mention_score=MentionScoresHead(self.bert_output_dim,self.score_method,self.max_mention_length)
        self.mention_loss=MentionLoss()
        self.ques_encoder=load_model(base_model_name,base_model_path)

    def forward(self,input_ids,input_mask,gold_mention_bounds,gold_mention_bounds_mask,device):

        input_ids=torch.LongTensor(input_ids).to(device)
        input_mask=torch.tensor(input_mask).to(device)
        input_mask_bool=input_mask.bool().to(device)
        token_type_ids=torch.ones(input_ids.size(),dtype=torch.int8).to(device)
        gold_mention_bounds=torch.clone(torch.tensor(gold_mention_bounds)).to(device)
        gold_mention_bounds_mask_bool=torch.tensor(gold_mention_bounds_mask).bool().to(device)
        bert_output,_=self.ques_encoder(input_ids,token_type_ids,input_mask)
        mention_scores,mention_bounds=self.mention_score(bert_output,input_mask_bool)   #forward(self, bert_output, input_mask)
        loss=self.mention_loss(gold_mention_bounds,gold_mention_bounds_mask_bool,mention_scores,mention_bounds)  #forward(self,gold_mention_bounds, gold_mention_bounds_mask, mention_logits, mention_bounds,):

        return loss,mention_scores,mention_bounds

    def predict(self,mention_scores,mention_bounds,top_k):
        # DIM:(bs,2), '2' for two indexes of max scores
        top_k_idx = mention_scores.topk(top_k)[1]
        #DIM:(bs,top_k,2)
        top_k_bounds = torch.zeros(mention_bounds.size(0), top_k, 2, dtype=torch.long)
        for i in range(mention_scores.size(0)):
            top_k_bounds[i, :, :] = mention_bounds[i, top_k_idx[i], :]

        return top_k_bounds