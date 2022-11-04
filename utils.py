import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import PIL
import matplotlib.pyplot as plt
import cv2

def set_seed(seednum,device):

    torch.manual_seed(seednum)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seednum)


class dataset(Dataset):
    def __init__(self, df,transform=None):
        super(dataset,self).__init__()
        self.df=df.reset_index(drop=True)
        self.transform=transform
    
    def __getitem__(self,index):
        
        if torch.is_tensor(index):
            index=index.tolist()

        data_df=self.df[index]

        image=cv2.imread(data_df)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image=self.transform(image)
        
        label=0

        if 'good' in data_df:
            label=1
        
        return image, label
    
    def __len__(self):

        return len(self.df)