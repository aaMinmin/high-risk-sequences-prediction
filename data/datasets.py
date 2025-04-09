
import os
import csv
Kmer=3
from functools import reduce
import pickle
Train_data=[]
Train_score=[]
with open(os.path.join('./Centers_result.csv'),'r',encoding='utf-8-sig') as f:
    r = csv.reader(f)
    for row in r:
        scores,seqs=row[0],row[1]
        while len(seqs)<150:
            seqs+='X'
        Train_data.append(seqs)
        Train_score.append(scores)

# with open(os.path.join('./5k.csv'),'r',encoding='utf-8-sig') as f2:
#     r = csv.reader(f2)
#     for row in r:
#         scores,seqs=row[0],row[1]
#         while len(seqs)<150:
#             seqs+='X'
#         Train_data.append(seqs)
#         Train_score.append(scores)

da = reduce(lambda x,y:[i+j for i in x for j in y],[['A','T','C','G','X']]*Kmer)
Train_DNAA = [[i[j:j+Kmer] for j in range(len(i)-Kmer+1)] for i in Train_data]
word_list = {w:i for i,w in enumerate(da,start=1)}
voc_size = len(word_list)
Traindata = [[word_list[n] for n in sen] for sen in Train_DNAA]
# list1=[['ATC','GTC','GAC','GTC','GAC','GTC','GAC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA','GGT','ACC','ACA'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC'],['ATC','GTC','GAC'],['ACA','GGT','ACC','TTC']]
# zzz=['4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111','2.111','3.111','4.111']
all=[]
for i,zz in zip(Traindata,Train_score):
# for i,zzz1 in zip(list1,zzz):
    q=[]
    for j in i:
        q.append(j)
    all.append([q,zz])
# all得到的是一整个序列的特征向量。
# traindataa = all[:int(0.95 * len(all))]
# testdataa = all[int(0.95 * len(all)):]
# with open(os.path.join("./", 'software-4kmer/traindata.pkl'), 'wb+') as f:
#     pickle.dump(traindataa, f)
with open(os.path.join("./", 'software-3kmer/testdata.pkl'), 'wb+') as f2:
    pickle.dump(all, f2)


