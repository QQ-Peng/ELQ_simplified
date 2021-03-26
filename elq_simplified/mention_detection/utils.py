def compute_precise_recall_overlap2(top_k_bounds,gold_mention_bounds):
    # 训练集中对mention_bounds的标注为包左不包右。且分词的结果并不包含[CLS],[SEP]
    num_mentions=get_gold_mention_num(gold_mention_bounds)
    gold_mention_bounds[:, :, 1] -= 1
    top_k_bounds=top_k_bounds.sort(dim=1)[0]
    """
    example:
    a=torch.tensor([[[2,3],[2,2],[4,5]],[[6,8],[6,7],[3,4]]])
    a=a.sort(dim=-1)[0]
    print(a)
        tensor([
        [[2, 2],
         [2, 3],
         [4, 5]],
        [[3, 4],
         [6, 7],
         [6, 8]]])
    """
    '''
		top_k_bounds: DIM:(bs, top_k,2)
		gold_mention_bounds: DIM:(bs,max_num_mention,2),include padding
	'''
    n = 0
    for i in range(top_k_bounds.size(0)):
        for j in range(top_k_bounds.size(1)):
            for m in range(gold_mention_bounds.size(1)):
                if gold_mention_bounds[i][m][0] <= 0:
                    continue
                if top_k_bounds[i, j, 0] < gold_mention_bounds[i, m, 0] and top_k_bounds[i, j, 1] < \
                        gold_mention_bounds[i, m, 0]:
                    continue
                elif top_k_bounds[i, j, 0] > gold_mention_bounds[i, m, 1] and top_k_bounds[i, j, 1] > \
                        gold_mention_bounds[i, m, 1]:
                    continue
                else:
                    n += 1
                    break #防止一个mention被重复计数
    pres=n / (top_k_bounds.size(0) * top_k_bounds.size(1))
    recall= n/num_mentions
    return pres,recall

def compute_precise_recall_overlap(top_k_bounds,gold_mention_bounds):
    #这个函数的会会剔除两种包含关系的bounds，比较复杂，并没有考虑到所有的情况仅供测试使用。
    # print('remove overlap')
    # 训练集中对mention_bounds的标注为包左不包右。且分词的结果并不包含[CLS],[SEP]
    num_mentions=get_gold_mention_num(gold_mention_bounds)
    gold_mention_bounds[:, :, 1] -= 1
    top_k_bounds=top_k_bounds.sort(dim=1)[0].tolist()
    top_k_bounds2 = []
    total_num = 0
    for bs in top_k_bounds:
        one_seq = []
        one_seq.append(bs[0])
        Len = len(bs)
        cur_len = 1
        for i in range(1, Len):
            if bs[i][0] == one_seq[cur_len - 1][0] and bs[i][1] > one_seq[cur_len - 1][1]:
                one_seq.pop()
                one_seq.append(bs[i])
            elif bs[i][1]==one_seq[cur_len - 1][1]:
                pass
            else:
                one_seq.append(bs[i])
                cur_len += 1
        total_num += cur_len
        top_k_bounds2.append(one_seq)

    """
    example:
    a=torch.tensor([[[2,3],[2,2],[4,5]],[[6,8],[6,7],[3,4]]])
    a=a.sort(dim=-1)[0]
    print(a)
        tensor([
        [[2, 2],
         [2, 3],
         [4, 5]],
        [[3, 4],
         [6, 7],
         [6, 8]]])
    """
    '''
		top_k_bounds: DIM:(bs, top_k,2)
		gold_mention_bounds: DIM:(bs,max_num_mention,2),include padding
	'''
    n = 0
    for i in range(len(top_k_bounds2)):
        for j in range(len(top_k_bounds2[i])):
            for m in range(gold_mention_bounds.size(1)):
                if gold_mention_bounds[i][m][0] <= 0:
                    continue
                if top_k_bounds2[i][j][0] < gold_mention_bounds[i, m, 0] and top_k_bounds2[i][j][1] < \
                        gold_mention_bounds[i, m, 0]:
                    continue
                elif top_k_bounds2[i][j][0] > gold_mention_bounds[i, m, 1] and top_k_bounds2[i][j][1] > \
                        gold_mention_bounds[i, m, 1]:
                    continue
                else:
                    n += 1
                    break #防止一个mention被重复计数
    pres=n / total_num
    recall= n/num_mentions
    return pres,recall,top_k_bounds2

def get_gold_mention_num(gold_mention_bounds):
    num_mentions=0
    for i in range(gold_mention_bounds.size(0)):
        for j in range(gold_mention_bounds.size(1)):
            if gold_mention_bounds[i,j,0]!=-1:
                num_mentions += 1
    return num_mentions



#exercise code, ignore it

# import torch
# top_k_bounds=torch.tensor([[[2,3],[2,2],[4,6],[5,7]],[[6,8],[6,7],[3,4],[4,5]]])
# top_k_bounds=top_k_bounds.sort(dim=1)[0].tolist()
#
# top_k_bounds2=[]
# total_num=0
# for bs in top_k_bounds:
#     one_seq=[]
#     one_seq.append(bs[0])
#     Len=len(bs)
#     cur_len=1
#     for i in range(1,Len):
#         if bs[i][0]==one_seq[cur_len-1][0] and bs[i][1]>one_seq[cur_len-1][1]:
#             one_seq.pop()
#             one_seq.append(bs[i])
#         else:
#             one_seq.append(bs[i])
#             cur_len+=1
#     total_num+=cur_len
#     top_k_bounds2.append(one_seq)




