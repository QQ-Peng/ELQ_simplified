# Date: 2020/11/28
# Author: Qianqian Peng

import random

def sample(infile_path,outfile_path,sample_percent=0.01,method='step'):
	'''

	:param file_path:
	:param sample_percent:
	:param method: step or random
	:return:
	'''
	if method=='step':
		step_n=int(1/sample_percent)
		infile=open(infile_path,'r')
		outfile=open(outfile_path,'w')
		line=infile.readline()
		n=0
		while line:
			if n%step_n==0:
				outfile.write(line)
			n+=1
			line=infile.readline()
		infile.close()
		outfile.close()
	elif method=='random':
		edge_len=sample_percent**0.5
		Min=0.5-edge_len/2
		Max=0.5+edge_len/2
		infile = open(infile_path, 'r')
		outfile = open(outfile_path, 'w',encoding='utf-8')
		line=infile.readline()
		while line:
			x=random.random()
			y=random.random()
			if x>=Min and x<=Max and y>=Min and y<=Max:
				outfile.write(line)
			line=infile.readline()
		infile.close()
		outfile.close()
	else:
		print("Only 'step' and 'random' methods are supported!")




infile_path='F:/Data/onepass/entity/entity.jsonl'
outfile_path='F:/Data/onepass/entity/sample_entity_rand_0.001.jsonl'

sample(infile_path,outfile_path,0.001,'random')
