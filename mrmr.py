#! /usr/bin/env python2.7

def main():
	fread=open('yuxi','r')
	content=fread.readlines()
	fout=open('1A1','w')
	num=0
	true_group=list()
	false_group=list()
	for line in content:
		line=line.strip()
		units=line.split('\t')
		class_set=units[1:12]
	#	print class_set
		data_set=units[13:]
		tmp=[]
		for x in range(len(data_set)):
			tmp.append('vector'+str(x))
		if num==0:
			fout.write('class'+','+','.join(tmp)+'\n')
		if class_set[0]=='1':
			true_group.append(class_set[1]+','+','.join(data_set))
		else:
			false_group.append(class_set[1]+','+','.join(data_set))
		num+=1
	num=0
	for line in true_group:
		fout.write(line+'\n')
		num+=1

	for line in false_group:
		if num==600:break
		fout.write(line+'\n')
		num+=1
	fout.close()
	fread.close()
	#	print data_set


if __name__ == '__main__':
	main()