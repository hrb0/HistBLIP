import glob
import os
import random
import json
import shutil



train_test_list = '/data3/Thanaporn/kmedicon/train_test.txt'

output_path = '/data3/Thanaporn/kmedicon/'

dir = '/data3/Thanaporn/kmedicon/WSIs'

f_w = open(os.path.join('/data3/Thanaporn/kmedicon/wsi_report_data.csv'),'w')
f_w.writelines('die,case_id,slide_id\n')

train_list_file = open(train_test_list, "r") 
data = train_list_file.read() 
train_list = data.split("\n")
train_list_file.close() 

train_anno = []
test_anno = []
for line in train_list:
    txt_name = line.split(',')[0]
    print(txt_name)
    patient_name = txt_name.split('.')[0]
    f_w.writelines(dir+','+patient_name+','+patient_name+'\n')
    




# txt_file = glob.glob(os.path.join(path,'*.txt'))
# num_all = len(txt_file)
# print(num_all)
# txt_file = [elem.split('/')[-1] for elem in txt_file]
# train_test = '/data3/Thanaporn/kmedicon/train_test.txt'
# train_list= random.choices(txt_file,k=int(0.8*num_all))
# test_list = [elem for elem in txt_file if elem not in train_list]
# f_w = open(train_test,'w')
# for patient_id in train_list:
#     f_w.writelines(patient_id+',train\n')
# for patient_id in test_list:
#     f_w.writelines(patient_id+',test\n')
# f_w.close()