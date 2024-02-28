import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    maxw, maxh, imgs, labels, label_length = 0, 0, [], [], []
    for item in batch:
        img, label_list = item
        maxh = max(img.shape[0], maxh)
        maxw = max(img.shape[1], maxw)
        imgs.append(img)

        label = np.zeros(260)
        for i in label_list:
            label[i] = 1.0
        labels.append(label)
        label_length.append(len(label_list))

    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (maxw, maxh))

    imgs = torch.tensor(np.array(imgs,dtype=np.float32)).permute(0, 3, 1, 2)
    lbs = torch.tensor(np.array(labels,dtype=np.float32))
    lls = torch.tensor(np.array(label_length))

    return imgs, lbs, lls


class MyDataset(Dataset):
    def __init__(self, train = False, test = False):
        self.items = []
        self.dictionary = {}
        if train:
            with open('Corel-5k/train.json', 'r') as f:
                train_dict = json.load(f)        
            
            for idx, word in enumerate(train_dict['labels']):
                self.dictionary[word] = idx
            
            for sample in train_dict['samples']:
                self.items.append( (f"Corel-5k/images/{sample['image_name']}", self.label_str2seq(sample['image_labels'])) )

            with open('Corel-5k/train_aug.json', 'r') as f:
                train_dict = json.load(f)

            for sample in train_dict['samples']:
                self.items.append( (f"Corel-5k/images/{sample['image_name']}", self.label_str2seq(sample['image_labels'])) )

        if test:
            with open('Corel-5k/test.json', 'r') as f:
                train_dict = json.load(f)
            
            for idx, word in enumerate(train_dict['labels']):
                self.dictionary[word] = idx

            for sample in train_dict['samples']:
                self.items.append( (f"Corel-5k/images/{sample['image_name']}", self.label_str2seq(sample['image_labels'])) )

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        path, label = self.items[index]
        if not os.path.exists(path):
            print(path)
            raise(f'path {path} not exists.')
        img = cv2.imread(path, 1)
        return img, label
    
    def label_str2seq(self, str_list):
        return list(map(lambda x:self.dictionary[x], str_list))



if __name__ == '__main__':
    dataset = MyDataset(test = True)
    #print(dataset[0])
    train_loader = torch.utils.data.DataLoader(dataset,
                                 batch_size=8,
                                 shuffle=False,
                                 drop_last=True,
                                 sampler=None, 
                                 batch_sampler=None, 
                                 collate_fn=collate_fn)
    for batch_id, (data,target,target_lengths) in enumerate(train_loader):
        pass