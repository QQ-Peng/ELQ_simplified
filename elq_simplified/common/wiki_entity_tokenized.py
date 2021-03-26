# Date: 2020/11/29
# Author: Qianqian Peng

#这个模块用来将wiki entity的text用bert-large进行分词，并进行长度统一化，
# 然后用blink的biencoder中的wiki-encoder(wiki-encoder是我提取出来的单个encoder)得到embeddings。

# import sys
# sys.path.append("/home/pqq/workdir/PYProject/myelq2")
from transformers import BertTokenizer
import json
import os
import torch
from blink_biencoder.wiki_encoder import WikiEncoderModule, load_para
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)



def tokenizer_wiki_entity(infile_path,outfile_path,tokenizer,max_len=128):
	if os.path.exists('outfile_path'):
		print(outfile_path,' exists! return None')
		return None
	infile = open(infile_path, 'r')
	all_entity_ids=[]
	n=1
	for line in infile:
		line=json.loads(line)
		text=line['text']
		if len(text)>2000:
			text=text[:2000]
		token_ids=tokenizer.encode(text)
		cur_len=len(token_ids)
		if cur_len>max_len:
			token_ids=token_ids[0:max_len-1]+[102]
			all_entity_ids.append(token_ids)
		elif cur_len<max_len:
			token_ids=token_ids+[0]*(max_len-cur_len)
			all_entity_ids.append(token_ids)
		else:
			all_entity_ids.append(token_ids)
		if n%50==0:
			print(n,' wikia entities tokenized.')
		n+=1
	all_entity_ids=torch.tensor(all_entity_ids)
	torch.save(all_entity_ids,outfile_path)
	infile.close()


def embed_wiki_entity(infile_path,outfile_path,wiki_encoder_path,para_path):
	#tokenizer: bert-base-uncased or bert-large-uncased
	with torch.no_grad():
		para=load_para(para_path)
		model = WikiEncoderModule(para)
		model.load_state_dict(torch.load(wiki_encoder_path))
		model.to('cuda')
		for i in model.parameters():
			i.requires_grad=False
		token_ids=torch.load(infile_path).to('cpu')

		all_embedding=[]
		n=0
		try:
			token_ids2 = torch.zeros(8, token_ids.size(1), dtype=token_ids.dtype, device='cpu')
			while n+8<token_ids.size(0):
				token_ids2=token_ids2.to('cpu')

				token_ids2[0:8,:]=token_ids[n:n+8,:]

				token_ids2=token_ids2.to('cuda')
				token_embeddings=model(token_ids2,None,None)
				token_embeddings=token_embeddings.to('cpu')
				all_embedding.append(token_embeddings)
				if n%64==0:
					print(n,' wiki entities embedded done.')
				n+=8
			token_ids2 = torch.zeros(1, token_ids.size(1), dtype=token_ids.dtype, device='cpu')
			while n<token_ids.size(0):
				token_ids2 = token_ids2.to('cpu')

				token_ids2[0,:]=token_ids[n,:]
				token_ids2=token_ids2.to('cuda')
				token_embeddings = model(token_ids2, None, None)
				token_embeddings=token_embeddings.to('cpu')
				all_embedding.append(token_embeddings)
				if n%64==0:
					print(n,' wiki entities embedded done.')
				n+=1
			torch.save(torch.cat(all_embedding), outfile_path)
		except:
			print('some error happen! save embeddings')
			torch.save(torch.cat(all_embedding),outfile_path)



if __name__=='__main__':
	tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')

	wiki_infile_path='/home/pqq/workdir/data/zeshel/documents/all.json'
	wiki_outfile_path='/home/pqq/workdir/PYProject/Data/wikia_token_ids.t7'


	#tokenizer_wiki_entity(wiki_infile_path,wiki_outfile_path,tokenizer)


	token_ids_infile_path='/home/pqq/workdir/PYProject/Data/wikia_token_ids.t7'
	embedding_outfile_path='/home/pqq/workdir/PYProject/Data/wikia_large_128.t7'

	wiki_encoder_path='../model/biencoder/wiki_encoder_large.bin'
	wiki_encoder_params_path='../model/biencoder/wiki_encoder_large.json'

	embed_wiki_entity(token_ids_infile_path,embedding_outfile_path,wiki_encoder_path,wiki_encoder_params_path)

	# embedding=torch.load('../Data/wiki_embedding.t7')