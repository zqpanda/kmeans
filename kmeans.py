#!/bin/env python2.7
import math,random

def dis_cal(a,b):
	sum=0
	for x in range(len(a)):
		if type(a[x]) == str and not a[x].isdigit():a[x]=0
		if type(b[x]) == str and not b[x].isdigit():b[x]=0
	dis=[(int(a[i])-int(b[i]))**2 for i in range(len(a))]
	for x in range(len(dis)):
		sum+=dis[x]
	return math.sqrt(sum)

def file_read(filename):
	fread=open(filename,'r')
	content=fread.readlines()
	id_list=list()
	value_list=list()
	for line in content:
		line=line.strip()
		units=line.split('\t')
		id=units[0]
		prop=units[1:49]
		test=units[49:]
		id_list.append(id)
		value_list.append({'Prop':prop,'Test':test})
	return dict(zip(id_list,value_list))


def main():
	chemical_dict=file_read('./chemical')
	sum=0
	for i in range(10000):
		a = chemical_dict[random.choice(chemical_dict.keys())]['Prop']
		b = chemical_dict[random.choice(chemical_dict.keys())]['Prop']
		score = dis_cal(a,b)
		sum += score
	target = sum/10000.0
	test_group=random.sample(chemical_dict.keys(),50)
	cyp={'1a1':[],'1a2':[],'1b1':[],'2a6':[],'2b6':[],'2c8':[],'2c9':[],'2c19':[],'2d6':[],'2e1':[],'3a4':[]}
	temp=cyp.keys()
	for unit in chemical_dict:
		if unit in test_group:continue
		for i in range(11):
			if chemical_dict[unit]['Test'][i]=='1':cyp[temp[i]].append(unit)
	cal_result={'1a1':[],'1a2':[],'1b1':[],'2a6':[],'2b6':[],'2c8':[],'2c9':[],'2c19':[],'2d6':[],'2e1':[],'3a4':[]}
	tp=预测为真，实际也为真
	fn=预测为假，实际为真
	fp=预测为真，实际为假
	tn=预测为假，实际也为假
	for member in cyp:
		size=len(cyp[member])
		for x in test_group:
			dis_sum=0
			value_x=chemical_dict[x]['Prop']
			for y in cyp[member]:
				value_y=chemical_dict[y]['Prop']
				dis_value=dis_cal(value_x,value_y)
				dis_sum+=dis_value
			mean_value=dis_sum/size
			if mean_value<=target:
				cal_result[member].append(x)
	real_result={'1a1':[],'1a2':[],'1b1':[],'2a6':[],'2b6':[],'2c8':[],'2c9':[],'2c19':[],'2d6':[],'2e1':[],'3a4':[]}
	for test in test_group:
		for i in range(11):
			if chemical_dict[test]['Test'][i]=='1':	real_result[temp[i]].append(test)
	final_sum=0
	for z in real_result:
		num=len(set(real_result[z]) & set(cal_result[z]))
		final_sum+=num
	total=0
	for z in real_result:
		total+=len(real_result[z])
	print final_sum,total




if __name__ == '__main__':
	main()



