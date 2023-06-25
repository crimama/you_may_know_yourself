import logging
import wandb
import time
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def cal_metrics(preds,targets):
    result = [] 
    acc = accuracy_score(targets,preds)
    rec = recall_score(targets,preds,average='macro')
    pre = precision_score(targets,preds,average='macro')
    f1 = f1_score(targets,preds,average='macro')
    return [acc,rec,pre,f1]

def anomaly_detection_evaluation(testloader,model):
    target_ams = [] 
    pred_ams = [] 
    target_labels = []
    model.eval() 
    for i, (imgs, gts, labels, anomaly_labels) in enumerate(testloader):
        with torch.no_grad():
            # Inference 
            outputs,(x1,x2,x3,x4) = model(imgs)
            # Average along channel 
            x2 = torch.mean(x2,dim=1,keepdim=True)
            x4 = torch.mean(x4,dim=1,keepdim=True)
            # Interpolatino to target size 
            x2 = F.interpolate(x2, size=224, mode='bilinear', align_corners=True)
            x4 = F.interpolate(x4, size=224, mode='bilinear', align_corners=True)
            # Normalize 
            x2 = torch.concat([(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in x2])
            x4 = torch.concat([(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in x4])
            # Calculate Anomaly Score 
            anomaly_map = torch.pow((x4 - x2),2)
            anomaly_map = torch.unsqueeze(anomaly_map,dim=1)
            # Anomaly map flatten keeping batch 
            pred_am = anomaly_map.flatten(1)
            target_am = gts.flatten(1).type(torch.int)
            
            target_ams.append(target_am.detach().cpu().numpy())
            pred_ams.append(pred_am.detach().cpu().numpy())
            
            target_labels.append(anomaly_labels.detach().cpu().numpy())
            
    target_ams = np.concatenate(target_ams)
    pred_ams = np.concatenate(pred_ams)
    
    target_labels = np.concatenate(target_labels)
    pred_labels = np.max(pred_ams,axis=1)
    
    pixel_auroc = roc_auc_score(target_ams.flatten(),pred_ams.flatten())
    img_auroc = roc_auc_score(target_labels.flatten(),pred_labels.flatten())
    return img_auroc, pixel_auroc 

def train(trainloader, model, criterion, accelerator, optimizer, cfg):
    total_preds = [] 
    total_targets = [] 
    total_loss = [] 
    model.train()
    for i, (imgs,labels) in enumerate(trainloader):
        
        #predict 
        outputs,_ = model(imgs)
        loss = criterion(outputs,labels) 
        accelerator.backward(loss)
        
        #loss update 
        optimizer.step()
        optimizer.zero_grad()
        
        total_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        total_targets.extend(labels.detach().cpu().numpy())
        total_loss.append(loss.detach().cpu().numpy())
        
        if cfg.BASE.multi_gpu:
            if torch.distributed.get_rank()==0:
                print(f"[{i}/{len(trainloader)}] | loss : {loss}")
        else:
            print(f"[{i}/{len(trainloader)}] | loss : {loss}")
            
    return total_preds,total_targets,np.mean(total_loss)


def test(testloader,model,criterion):
    test_preds = [] 
    test_targets = [] 
    test_loss = [] 
    model.eval() 
    for i, (imgs, gts, labels, anomaly_labels) in enumerate(testloader):
        with torch.no_grad():
            outputs,(x1,x2,x3,x4) = model(imgs)
            loss = criterion(outputs,labels)
            
            test_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            test_targets.extend(labels.detach().cpu().numpy())
            test_loss.append(loss.detach().cpu().numpy())
            
    return test_preds,test_targets,np.mean(test_loss)

def fit(model, 
        trainloader, validloader, testloader, 
        optimizer,scheduler, criterion, 
        accelerator, wandb,
        cfg):

    model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
                                                                    model, optimizer, trainloader, validloader, testloader, scheduler
                                                                    )
    
    best_valid_f1 = 0
    for epoch in range(cfg['TRAIN']['epochs']):
        if cfg.BASE.multi_gpu:
            if torch.distributed.get_rank()==0:
                print(f"Epoch : {epoch}/{cfg['TRAIN']['epochs']} Train start")
        else:
            print(f"Epoch : {epoch}/{cfg['TRAIN']['epochs']} Train start")
           
        # Train and Train evaluation          
        train_preds, train_targets, total_loss  = train(trainloader,model,criterion,accelerator,optimizer,cfg)
        [train_acc,train_rec,train_pre,train_f1] = cal_metrics(train_preds, train_targets)
        
        if cfg.BASE.multi_gpu:
            if torch.distributed.get_rank()==0:
                print(f"Epoch : {epoch} Evaluation start")
        else:
            print(f"Epoch : {epoch} Evaluation start")
        
        # Test Evaluation 
        test_preds, test_targets, test_loss  = test(testloader,model,criterion)
        [test_acc,test_rec,test_pre,test_f1] = cal_metrics(test_preds, test_targets)
        
        # Anomaly Evaluation 
        img_auroc, pixel_auroc = anomaly_detection_evaluation(testloader,model)
        
        
        if cfg.BASE.multi_gpu:
            if torch.distributed.get_rank()==0:
                print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Train loss : {total_loss} | Train F1 : {train_f1} | Train ACC : {train_acc} | Train Recall : {train_rec} | Train Precision : {train_pre}")
                print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Test loss : {test_loss} | Test F1 : {test_f1} | Test ACC : {test_acc} | Test Recall : {test_rec} | Test Precision : {test_pre}")
                print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Image AUROC : {img_auroc} | Pixel AUROC : {pixel_auroc}")
        else:
            print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Train loss : {total_loss} | Train F1 : {train_f1} | Train ACC : {train_acc} | Train Recall : {train_rec} | Train Precision : {train_pre}")
            print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Test loss : {test_loss} | Test F1 : {test_f1} | Test ACC : {test_acc} | Test Recall : {test_rec} | Test Precision : {test_pre}")
            print(f"[{epoch}/{cfg['TRAIN']['epochs']}] | Image AUROC : {img_auroc} | Pixel AUROC : {pixel_auroc}")
            
        #wandb logging 
        if cfg.TRAIN.wandb.use:
            log = {
                    'train_loss' : total_loss,
                    'train_Accuracy' : train_acc,
                    'train_Recall' : train_rec,
                    'train_Precision' : train_pre,
                    'train_F1' : train_f1,
                    'test_loss' : test_loss,
                    'test_Accuracy' : test_acc,
                    'test_Recall' : test_rec,
                    'test_Precision' : test_pre,
                    'test_F1' : test_f1,     
                    'Image_auroc' : img_auroc,
                    'Pixel_auroc' : pixel_auroc                
                    }
            if cfg.BASE.multi_gpu:
                if torch.distributed.get_rank()==0:
                    wandb.log(log)
            else:
                wandb.log(log)
                            
        # model save 
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            {
            "model": unwrapped_model.state_dict()
            },
            os.path.join(cfg.BASE.save_dir,'model.pt')
            )
        
        
            
        