#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-
import numpy as np
import sys
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
		prop=chem_dict[line]['Prop']
		result=chem_dict[line]['Result']
		for x in vectorsNeeded:
			vector_mrmr.append(float(prop[int(x)]))
		group.append(vector_mrmr)
		labels.append(int(result[chosen_label]))
	group = np.array(group)
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
		prop = [float(x) for x in prop]
		id_list.append(chem_id)
		value_list.append({'Prop':prop,'Result':result})
	return dict(zip(id_list,value_list))


def classify(newInput,dataSet,labels,k):
	'''
	knn分类算法
	'''
	#第一步欧几里得距离计算
	numSamples = dataSet.shape[0]
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
	if maxIndex=='1':
		return round(float(maxCount)/float(k),2)
	else:
		return round(1 - float(maxCount)/float(k),2)

def analysis(real,predict):
	diff = list(np.array(predict) - np.array(real))
	pre_p = predict.count(1)
	pre_n = predict.count(0)
	fp_num = diff.count(1)
	fn_num = diff.count(-1)
	tp_num = pre_p - fp_num
	tn_num = pre_n - fn_num
	return float(fp_num)/float(fp_num+tn_num),float(tp_num)/float(tp_num+fn_num)

def normalize(chem_dict):
	temp_list=[]
	for (name,prop) in chem_dict.items():
		temp_list.append(prop['Prop'])
	all_prop=np.array(temp_list)
	trans_prop=np.transpose(all_prop)
	temp=[]
	for prop_set in trans_prop:
		mean=np.mean(prop_set)
		var=np.var(prop_set)**0.5
		temp.append((mean,var))
	normalized_dict={}
	for (name,prop) in chem_dict.items():
		tmp=[ (prop['Prop'][i] - temp[i][0])/float(temp[i][1]) for i in range(len(prop['Prop']))]
		normalized_dict[name]={'Result':prop['Result'],'Prop':tmp}
	return normalized_dict

def mrmr_format(chem_dict,cyp_family):
	title = 'class'+','+','.join(['vector' + str(num) for num in range(498)])
	for cyp in cyp_family:
		fout=open('./'+cyp,'w')
		fout.write(title+'\n')
		pos=cyp_family.index(cyp)
		p_group=[]
		n_group=[]
		for (name,prop) in chem_dict.items():
			if prop['Result'][pos]=='1':p_group.append(name)
			else:n_group.append(name)
		num=len(p_group)
		if len(p_group)>len(n_group):
			num=len(n_group)
		for flag in range(num):
			p_name=p_group[flag]
			fout.write('1'+','+','.join([str(round(x,2)) for x in chem_dict[p_name]['Prop']])+'\n')
			n_name=n_group[flag]
			fout.write('0'+','+','.join([str(round(y,2)) for y in chem_dict[n_name]['Prop']])+'\n')
		print cyp+' is Done!'
		fout.close()
		


def main():
	'''
	主程序
	'''
	cyp_family = ['1a1','1a2','1b1','2a6','2b6','2c8','2c9','2c19','2d6','2e1','3a4']
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
	mrmr_format(normalize(chem_dict),cyp_family)
	'''
	cyp_family = ['1a1','1a2','1b1','2a6','2b6','2c8','2c9','2c19','2d6','2e1','3a4']
#	cyp_roc = dict()
	for member in cyp_family:
		fout=open(member,'w')
		roc_data = list()
		dataSet,labels=createDataSet('./train',chem_dict,member)
		testSet,test_labels=createDataSet('./test',chem_dict,member)
		for k in range(1,21):
			for step in np.arange(0.01,1,0.01):
				predict_result = list()
				for sample in testSet:
					score=classify(sample,dataSet,labels,k)
					if score>=step:
						predict_result.append('1')
					else:
						predict_result.append('0')
					fpr,tpr = analysis(test_labels,predict_result)
					fout.write(str(fpr)+'\t'+str(tpr)+'\n')
				fout.write(str(step)+'************************'+'\n')
			fout.write('K='+str(k)+'\n')
		fout.close()
#		cyp_roc[member]=roc_data
#	print cyp_roc
	'''

if __name__ == '__main__':
	main()
