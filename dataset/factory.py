from .build import *
import os 
from torch.utils.data import DataLoader 

def load_dataset(dataset_name:str, dataset_dir:str,class_name:str, img_size:int, batch_size:int,**params):
    if dataset_name.lower() in ['mvtecad','mvtedad_loco']:
        trainset = TrainDataset(
                    dataset_name = dataset_name,
                    dataset_dir  = dataset_dir,
                    class_name   = class_name,
                    img_size     = img_size
                    )
        testset = TestDataset(
                    dataset_name = dataset_name,
                    dataset_dir  = dataset_dir,
                    class_name   = class_name,
                    img_size     = img_size
                    )
    elif dataset_name.lower() == 'visa':
        trainset = ViSADataset(
                    dataset_dir  = os.path.join(dataset_dir,'ViSA'),
                    class_name   = class_name,
                    img_size     = img_size,
                    mode         = params['mode']
        )
        testset = ViSADataset(
                    dataset_dir  = os.path.join(dataset_dir,'ViSA'),
                    class_name   = class_name,
                    img_size     = img_size,
                    mode         = params['mode'],
                    train        = False 
        )
        
    return trainset,testset 

def load_dataloader(trainset,validset,testset,batch_size,shuffle=True):
    trainloader = DataLoader(trainset,
                             batch_size = batch_size,
                             shuffle = shuffle)
    validloader = DataLoader(validset,
                            batch_size = batch_size,
                            shuffle = False)
    
    testloader = DataLoader(testset,
                            batch_size = batch_size,
                            shuffle = False)
    
    return trainloader,validloader,testloader