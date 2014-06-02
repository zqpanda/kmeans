#!/bin/env python2.7
# -*- coding:utf-8 -*-
import numpy as np
import random,operator

def createDataSet(trainpath,chem_dict,family):
	'''
	构造数据
	'''
	family_labels = ['1a1','1a2','1b1','2a6','2b6','2c8','2c9','2c19','2d6','2e1','3a4']
	chosen_label = family_labels.index(family)
	vectorsNeeded = vectors(family)
	fread=open(trainpath,'r')
	content=fread.readlines()
	group=list()
	labels=list()
	for line in content:
		line=line.strip()
		vector_mrmr=list()
		for i in vectorsNeeded:
			vector_mrmr.append(chem_dict[line]['Prop'][int(i)])
		group.append(vector_mrmr)
		labels.append(chem_dict[line]['Result'][chosen_label])
	print group	
	group = np.array([group])
	return group,labels


def vectors(family):
	'''
	读取mrmr特征筛选结果
	'''
	vector_path='../kmeans/'+family.upper()+'_vector'
	fread=open(vector_path,'r')
	content=fread.readlines()
	vector_index=list()
	for line in content:
		line=line.strip('vector\n ')
		if line not in vector_index:
			vector_index.append(line)
	return vector_index


def chemDict(filepath):
	fread = open(filepath,'r')
	content = fread.readlines()
	id_list = list()
	value_list = list()
	for line in content:
		line = line.strip()
		if line.startswith('number'):continue
		units = line.split('\t')
		chem_id = units[0]
		result = units[1:12]
		prop = units[13:]
		id_list.append(chem_id)
		value_list.append({'Prop':prop,'Result':result})
	return dict(zip(id_list,value_list))


def classify(newInput,dataSet,labels,k):
	'''
	knn分类算法
	'''
	#第一步欧几里得距离计算
	numSamples = dataSet.shape[0]
	print numSamples
	diff=np.tile(newInput,(numSamples,1)) - dataSet
	squaredDiff = diff ** 2
	squaredDist = np.sum(squaredDiff,axis=1)
	distance = squaredDist ** 0.5

	#第二步，距离排序
	sortedDistIndices = np.argsort(distance)
	classCount = {}
	for i in xrange(k):
		voteLabel = labels[sortedDistIndices[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1
	#第三步筛选出出现数量最多的标签并返回
	maxCount = 0
	for key,value in classCount.items():
		if value > maxCount:
			maxCount = value
			maxIndex = key
	return maxIndex

def main():
	'''
	主程序
	'''
	chem_dict = chemDict('../kmeans/yuxi')
	'''
	#测试组、训练组划分，测试集100
	testX=open('test','w')
	trainningX=open('train','w')
	test_group = random.sample(chem_dict.keys(),100)
	for x in chem_dict:
		if x in test_group:
			testX.write(x+'\n')
		else:
			trainningX.write(x+'\n')
	testX.close()
	trainningX.close()
	'''
	dataSet,labels=createDataSet('./train',chem_dict,'1a1')
	testSet,test_labels=createDataSet('./test',chem_dict,'1a1')
	testX=testSet[0][0]
#	print testX
#	k=3
#	outputLabel=classify(testX,dataSet,labels,k);
#	print outputLabel

if __name__ == '__main__':
	main()