from torch.utils.data import Dataset,DataLoader 
from torchvision import transforms 
from PIL import Image 
import torch 
from glob import glob 
import numpy as np 
import pandas as pd 
from torchvision.datasets import ImageFolder
import os 


class TrainDataset(Dataset):
    def __init__(self, dataset_name:str, dataset_dir:str , class_name:str, img_size:int):
        if class_name == 'all':
            self.img_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'*/train/*/*.png')))
        else: 
            self.img_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'{class_name}/train/*/*.png')))
        self.labels = pd.Series(self.img_dirs).apply(lambda x: x.split('/')[-4]).unique()
        self.l2i = {c:n for n,c in enumerate(self.labels)}
        self.i2l = {n:c for n,c in enumerate(self.labels)}
        
        self.img_size = img_size 
        self.transform = self.create_basic_transform()
        
    def create_basic_transform(self):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        return transform 
    
    def get_model_transform(self,transform):
        self.transform = transform 
        
    def __len__(self):
        return len(self.img_dirs)
    
    def img_load(self,img_dir):
        img = Image.open(img_dir).convert('RGB')
        img = self.transform(img)
        return img 
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        img_label = self.l2i[img_dir.split('/')[-4]]
        
        img = self.img_load(img_dir)
        
        return img,img_label 
    
class TestDataset(TrainDataset):
    def __init__(self, dataset_name:str, dataset_dir:str , class_name:str, img_size:int):
        if class_name == 'all':
            self.img_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'*/test/*/*.png')))
            self.gt_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'*/ground_truth/*/*.png')))
        else: 
            self.img_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'{class_name}/test/*/*.png')))
            self.gt_dirs = sorted(glob(os.path.join(dataset_dir,dataset_name,f'{class_name}/ground_truth/*/*.png')))
            
        self.labels = pd.Series(self.img_dirs).apply(lambda x: x.split('/')[-4]).unique()
        self.l2i = {c:n for n,c in enumerate(self.labels)}
        self.i2l = {n:c for n,c in enumerate(self.labels)}
        
        
        self.img_size = img_size 
        self.transform = self.create_basic_transform()
        self.gt_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((self.img_size,self.img_size))])
        
    def gt_load(self,img_dir):
        if img_dir.split('/')[-2] == 'good':
            gt = torch.zeros(1,self.img_size,self.img_size)
        else:
            gt_dir = (img_dir[:-4] + '_mask' + img_dir[-4:]).replace('test','ground_truth')
            gt = Image.open(gt_dir).convert('L')
            gt = self.gt_transform(gt)
        return gt 
        
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        
        img_label = self.l2i[img_dir.split('/')[-4]]
        anomaly_label = 0 if self.img_dirs[idx].split('/')[-2]=='good' else 1 
        
        gt = self.gt_load(img_dir)
        img = self.img_load(img_dir)
        
        return img, gt, img_label, anomaly_label
    
class ViSADataset(TrainDataset):
    def __init__(self,dataset_dir,img_size,class_name = 'candle',mode='full',train=True):
        
        self.dataset_dir     = dataset_dir                   # Dataset directory 
        self.img_size        = img_size 
        self.mode            = mode                          # Training mode : Fullshot, 2cls Fewshot, 2cls Highshot 
        self.img_cls         = class_name                    # Image Class 
        
        
        
        self.df = self._read_csv(mode)               # Load df containing information of img and mask 
        self._load_dirs(class_name,train)               # Following df, load directorys of imgs and masks 
        self.transform  = self.create_basic_transform()
        self.gt_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((self.img_size,self.img_size))])
        
        self.train = train  
        self.l2i = {c:i for i,c in  enumerate(pd.Series(self.img_dirs).apply(lambda x : x.split('/')[0]).unique())}
            
    def _read_csv(self,mode):
        # Choose a mode of Training : Fullshot, 2cls Fewshot, 2cls Highshot 
        if mode == 'full':
            df = pd.read_csv(os.path.join(self.dataset_dir,'split_csv','1cls.csv'))
        elif mode == 'fewshot':
            df = pd.read_csv(os.path.join(self.dataset_dir,'split_csv','2cls_fewshot.csv'))
        elif mode == 'highshot':
            df = pd.read_csv(os.path.join(self.dataset_dir,'split_csv','2cls_highshot.csv'))
        return df 
    
    def _load_dirs(self,img_cls,train):
        # Choose either Training or Test and additionaly Class of img (ex : Candle)
        if img_cls == 'all': # In case using All type of Image Claases 
            if train:
                self.img_dirs = self.df[self.df['split']=='train']['image'].values
                self.gt_dirs = self.df[self.df['split']=='train']['mask'].values
            else:
                self.img_dirs = self.df[self.df['split']=='test']['image'].values
                self.gt_dirs = self.df[self.df['split']=='test']['mask'].values
        else: # In case only using one class of image 
            if train:
                self.img_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['image'].values
                self.gt_dirs = self.df[(self.df['split']=='train') & (self.df['object'] == self.img_cls)]['mask'].values
            else:
                self.img_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['image'].values
                self.gt_dirs = self.df[(self.df['split']=='test') & (self.df['object'] == self.img_cls)]['mask'].values
            

    def gt_load(self,gt_dir):
        if str(gt_dir) == 'nan': #Normal Case
            gt = torch.zeros(3,self.img_size,self.img_size)
        else:
            gt = Image.open(os.path.join(self.dataset_dir,gt_dir)).convert('RGB')
            gt = self.gt_transform(gt)
        return gt 
        
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        img = self.img_load(os.path.join(self.dataset_dir,img_dir))
        img_label = self.l2i[self.img_dirs[idx].split('/')[0]]
        
        if self.train:
            return img, img_label 
        
        else: # test mode 
            gt_dir = self.gt_dirs[idx]
            gt = self.gt_load(gt_dir)
            anomaly_label = 0 if self.img_dirs[idx].split('/')[-2] == 'Normal' else 1
            return img, gt, img_label, anomaly_label
            