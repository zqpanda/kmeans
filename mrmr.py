#! /usr/bin/env python2.7

def main():
	fread=open('yuxi','r')
	content=fread.readlines()
	fout=open('1A1','w')
	num=0
	for line in content:
		line=line.strip()
		units=line.split('\t')
		class_set=units[1:12]
	#	print class_set
		data_set=units[13:]
		tmp=[]
		for x in range(len(data_set)):
			tmp.append(str(x))
		if num==0:
			fout.write('class'+','+','.join(tmp)+'\n')
		else:
			fout.write(class_set[0]+','+','.join(data_set)+'\n')
		num+=1
	fout.close()
	fread.close()
	#	print data_set


if __name__ == '__main__':
	main()