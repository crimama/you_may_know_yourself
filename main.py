import yaml 
import torch 
import numpy as np 
import random 
import os 
from accelerate import Accelerator

from train import fit 
from dataset import load_dataset, load_dataloader 
from models import load_model 
from arguments import parser
import wandb
import warnings
warnings.filterwarnings("ignore")

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    

def main(cfg):
    #Seed Setting 
    torch_seed(cfg['TRAIN']['seed'])
    
    #Load Dataset & DataLoader
    trainset,testset = load_dataset(
                                        dataset_name = cfg['DATA']['dataset_name'],
                                        dataset_dir  = cfg['DATA']['dataset_dir'],
                                        class_name   = cfg['DATA']['class_name'],
                                        img_size     = cfg['DATA']['img_size'],
                                        batch_size   = cfg['DATA']['batch_size'],
                                        **{'mode':'full'}
                                    )
    trainloader,validloader,testloader = load_dataloader(trainset,testset,testset,
                                                        batch_size = cfg['DATA']['batch_size'],
                                                        shuffle=True)
        
    #Load model 
    model = load_model(
                        model_name = cfg['MODEL']['model_name'],
                        pretrained = cfg['MODEL']['pretrained'],
                        num_class  = len(trainset.l2i),
                        )
    transform = model.get_model_transform()
    
    trainloader.dataset.get_model_transform(transform)
    
    # criterion 
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['TRAIN']['optimizer']](model.parameters(), lr=cfg['TRAIN']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['TRAIN']['epochs'], T_mult=1, eta_min=0.00001)
    criterion = torch.nn.CrossEntropyLoss()

    accelerator = Accelerator(mixed_precision = 'fp16')
    
    if cfg.TRAIN.wandb.use:
        if cfg.BASE.multi_gpu:
            if torch.distributed.get_rank()==0:
                wandb.init(project=cfg.TRAIN.wandb.project_name,name='/'.join(cfg.BASE.save_dir.split('/')[2:]), config=cfg)
        else:
            wandb.init(project=cfg.TRAIN.wandb.project_name,name='/'.join(cfg.BASE.save_dir.split('/')[2:]), config=cfg)        
    fit(
        model = model,
        trainloader = trainloader,
        validloader = validloader,
        testloader  = testloader,
        optimizer   = optimizer,
        scheduler   = scheduler,
        criterion   = criterion,
        accelerator = accelerator,
        wandb       = wandb, 
        cfg      = cfg
    )

if __name__ == "__main__":
    
    cfg = parser()
    #yaml.dump(cfg, open(os.path.join(cfg.BASE.save_dir, 'config.yaml'), 'w'))    
    main(cfg)