# Date: 2020/12/6
# Author: Qianqian Peng

#写此代码的原因：
#我发现采用random的方式采集wiki样本，在mention detection和entity linking（采用hard negative）同时进行的
#方法中，EL的precision为0，因此我怀疑random的hard negative不够hard，写这个代码的目的就是通过计算训练数据
#与五百多万entity的相似度，为条训练数据找出10条最相似的entity。

from blink_biencoder.wiki_encoder import WikiEncoderModule


