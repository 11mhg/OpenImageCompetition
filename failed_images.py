import csv
fail_index = set()
with open('./Pipeline/index_train.txt','r') as f:
    for i in f:
        fail_index.add(i.strip())
        
with open('/home/lam/Desktop/Dataset/OpenImage/annotations/train-bbox.csv','r') as f:
    with open('./Pipeline/failed_images.csv','a') as p:
        bbox_reader = csv.reader(f,delimiter=',')
        next(bbox_reader)
        for i in bbox_reader:
            if i[0] in fail_index:
                p.write(str(i)+'\n')
