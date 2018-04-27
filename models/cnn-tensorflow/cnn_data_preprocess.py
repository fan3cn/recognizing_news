# coding: utf-8

dataset_info=[]
content_matters=[]
training=[]
val=[]
test=[]
i=0

path = "../../data/"
with open(path+'News_pic_label_train.txt','r') as labelset:
  for line in labelset:
    items=line.split('\t')
    label=int(items[1])
    content_matters.append(items[3])
    dataset_info.append(label)

with open(path+'News_info_train_seg.txt','r') as dataset:
  for line in dataset:
    items = line.split('\t')
    content = items[1]
    words=content.split(' ')
    label=dataset_info[i]
    l=str(label)
    
    if(label==1 and content_matters[i].find('NULL')>=0):
        #print(l+' '+content_matters[i])
        i+=1
        continue
    
    if(i<30000): training.append([l,' '.join(words)])
    else: val.append([l,' '.join(words)])
    i=i+1

with open(path+'News_info_validate_seg.txt','r') as dataset:
  for line in dataset:
    items = line.split('\t')
    content = items[1]
    words=content.split(' ')
    test.append(' '.join(words))

with open('data/train.txt','w') as fout:
  for line in training:
    fout.write('%s \n' % '\t'.join(line))

with open('data/val.txt','w') as fout:
  for line in val:
    fout.write('%s \n' % '\t'.join(line))

with open('data/test.txt','w') as fout:
  for line in test:
    fout.write('%s \n' % str(line))


