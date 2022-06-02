import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import imageio as io


class MyData(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transforms=None):
        self.root = root
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        #comment
    def __getitem__(self, idx):
        current_name = 'image_0.5_0.05_{}'.format(idx)
        
        trigger = 0
        boxes = []
        label = []
        
        for i in range(len(self.data)):
            if idx == len(self.data):
                img_name = os.path.join(self.root, self.data.iloc[idx-1, 1])
                image = io.imread(img_name + '.jpg')
                break
            if current_name == self.data.iloc[idx, 1]:
                x1 = self.data.iloc[idx, 3] - self.data.iloc[idx, 5]
                y1 = self.data.iloc[idx, 4] - self.data.iloc[idx, 5]
                x2 = self.data.iloc[idx, 3] + self.data.iloc[idx, 5]
                y2 = self.data.iloc[idx, 4] + self.data.iloc[idx, 5]
                boxes.append([x1, y1, x2, y2])
                label.append(self.data.iloc[idx, 6])
                idx = idx + 1
                trigger = 1
            elif trigger == 1:
                img_name = os.path.join(self.root, self.data.iloc[idx-1, 1])
                image = io.imread(img_name + '.jpg')
                break
            else:
                idx += 1

        target = {self.data.iloc[idx-1, 1]: image, 'boxes': boxes, 'label': label}
        if self.transforms is not None:
            target = self.transforms(target)
        return target
        

    def __len__(self):
        return 20

my_data = MyData('my_data/sigma_0.5/noise_0.05/data_sigma_0.5_noise_0.05.csv', "my_data/sigma_0.5/noise_0.05")
for i in range(len(my_data)):
    print(my_data[i])