import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.model = args.model

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            if args.model == "Histgen": 
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            else: 
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'], return_tensors="pt")
                self.examples[i]['ids'] = self.examples[i]['ids']['input_ids'][0]

            #* Below is the code to generate the mask for the report
            #* such a mask is used to indicate the positions of actual tokens versus padding positions in a sequence.
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids']) 

    def __len__(self):
        return len(self.examples)
    
class PathologySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = os.path.join(self.image_dir, image_id + '.pt')
        image = torch.load(image_path)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        if self.model == 'BLIP':
            report_ids = torch.where((report_ids == 101) , self.tokenizer.bos_token_id, report_ids)
            report_ids = torch.where((report_ids == 102) , self.tokenizer.eos_token_id, report_ids)
        
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample