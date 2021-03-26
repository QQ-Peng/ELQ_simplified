# Date: 2020/11/30
# Author: Qianqian Peng

#这个模块用来将训练数据集中linking的entity与随机采样的0.01的wiki_KB进行整合。
#电脑内存太小无法对整个wiki进行搜索才会用该模块。
import json
import time


def merge_train_sample(trainfile_path,sampleKB_path,KB_merge_outpath,train_merge_outpath):
	start_time=time.time()
	trainfile=open(trainfile_path,"r")
	sampleKB=open(sampleKB_path,"r")
	train_entity=[]
	for line in trainfile:
		line=json.loads(line)
		for e in line["entity"]:
			train_entity.append(e)
	trainfile.close()

	reserved_samoleKB=[]
	for line in sampleKB:
		line2=json.loads(line)
		if line2["entity"] in train_entity:
			continue
		else:
			reserved_samoleKB.append(line)
	sampleKB.close()

	print("Remove Duplicates Done")

	trainfile = open(trainfile_path, "r")
	train_merge=open(train_merge_path,"w",encoding="utf-8",newline="\n")
	final_sampleKB=[]
	sample_id=0
	for line in trainfile:
		line=json.loads(line)
		line["sample_id"]=[]
		for i in range(len(line["entity"])):
			e={"text":line["label"][i],"idx":"in source KB","title":"in source KB","entity":line["entity"][i],"sample_id":sample_id}
			final_sampleKB.append(json.dumps(e))
			line["sample_id"].append(sample_id)
			sample_id+=1
		train_merge.write(json.dumps(line)+"\n")
	trainfile.close()
	train_merge.close()
	print("Add train entity Done! Write merged train file Done")

	for line in reserved_samoleKB:
		line=json.loads(line)
		line["sample_id"]=sample_id
		final_sampleKB.append(json.dumps(line))
		sample_id+=1

	print("Add sampleKB Done")

	outfile=open(KB_merge_outpath,"w",encoding="utf-8",newline="\n")
	for line in final_sampleKB:
		outfile.write(line+"\n")
	outfile.close()
	end_time=time.time()
	print("time cost: ",end_time-start_time)


trainfile_path = "../Data/train.jsonl"
sampleKB_path = "../Data/sample_entity_rand.jsonl"
finalKB_path="../Data/train_sampleKB_merge.jsonl"
train_merge_path="../Data/train_merge.jsonl"

merge_train_sample(trainfile_path,sampleKB_path,finalKB_path,train_merge_path)