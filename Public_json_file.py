import glob
import os
import random
import json
import shutil
import re

path = '/data2/yumi/kmedicon/patch_pu/pt_files/'

# train_test_list = '/data3/Thanaporn/kmedicon/train_test.txt'

output_path = '/data3/Thanaporn/kmedicon/'

if not os.path.exists(output_path):
    os.makedirs(output_path)


test_anno = []

for txt_name in os.listdir(path):
    data_anno = dict(
        id = txt_name.split('.')[0],
        report = ' ',
        image_path = ' ',
        split = 'val')
    test_anno.append(data_anno)

json_annotation = dict(
    train = [],
    val = test_anno,
    test = test_anno,
)

output_name = output_path+'/annotation_public_dummy.json'
with open(output_name,'w') as outfile:
    json.dump(json_annotation, outfile)



# txt_file = glob.glob(os.path.join(path,'*.txt'))
# txt_file = [elem.split('/')[-1] for elem in txt_file]
# num_all = len(txt_file)
# print(num_all)
# train_test = '/data3/Thanaporn/kmedicon/train_test.txt'
# train_list= random.sample(txt_file,k=int(0.8*num_all))
# test_list = [elem for elem in txt_file if elem not in train_list]
# print(len(train_list),' + ',len(test_list))
# f_w = open(train_test,'w')
# for patient_id in train_list:
#     f_w.writelines(patient_id+',train\n')
# for patient_id in test_list:
#     f_w.writelines(patient_id+',test\n')
# f_w.close()