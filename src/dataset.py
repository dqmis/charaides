"""
Module that defines custom PyTorch Datasets to load and process data.
"""
import ast
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class DoodleDataset(Dataset):
    
    def draw_it(self, strokes):
        image = Image.new("P", (256, 256), color=255)
        image_draw = ImageDraw.Draw(image)
        for stroke in ast.literal_eval(strokes):
            for i in range(len(stroke[0])-1):
                image_draw.line([stroke[0][i], 
                                 stroke[1][i],
                                 stroke[0][i+1], 
                                 stroke[1][i+1]],
                                fill=0, width=5)
        return image.convert('RGB')
    
    def get_df(self, source_dir, class_list, im_per_class):
        li = []
        for class_ in class_list:
            df = pd.read_csv('{}/{}'.format(source_dir, class_),
                             engine='python',
                             usecols=['drawing', 'recognized', 'word'],
                             nrows=im_per_class*5//4)
            df = df[df.recognized == True][['drawing', 'word']]
            li.append(df.head(im_per_class))
        return pd.concat(li, axis=0, ignore_index=True)
    
    def __init__(self, source_dir, class_list=[], im_size=224, im_per_class=3000, transform=None):
        self.transform = transform
        self.im_size = im_size
        self.source_dir = source_dir
        if len(class_list) < 1:
            self.class_list = os.listdir(self.source_dir)
        else:
            self.class_list = class_list
        self.source_df = self.get_df(self.source_dir, self.class_list, im_per_class)
        self.one_hot_labels = pd.get_dummies(self.source_df['word'])
        
    def get_labels(self):
        return self.source_df.word.unique()
    
    def export_labels(self):
        file = open('class_names.txt', 'w')
        for name in [name.replace(' ', '_') for name in self.source_df.word.unique()]:
            file.write('{}\n'.format(name))
        file.close()

    def __len__(self):
        return self.source_df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_name = self.source_df.iloc[idx, 1]
        label = torch.LongTensor(self.one_hot_labels.iloc[idx].values)
        image = self.draw_it(self.source_df.iloc[idx, 0])
        sample = {'image': image, 'label': label, 'class_name': class_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
