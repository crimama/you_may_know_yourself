
import timm 
import torch.nn as nn 
from torchvision import transforms

class BaseModel(nn.Module):
    def __init__(self,model_name,pretrained=True, num_class = 15 ):
        super(BaseModel,self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_class)
        
        self.init_layer = nn.Sequential(*list(self.model.children())[:4])
        self.block1 = nn.Sequential(*list(self.model.children())[4])
        self.block2 = nn.Sequential(*list(self.model.children())[5])
        self.block3 = nn.Sequential(*list(self.model.children())[6])
        self.block4 = nn.Sequential(*list(self.model.children())[7])
        self.head = nn.Sequential(*list(self.model.children())[8:])
        
    def forward_block(self,x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1,x2,x3,x4
    
    def get_model_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop((224,224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return transform 
        
    def forward(self,x):
        x = self.init_layer(x)
        x1,x2,x3,x4 = self.forward_block(x)
        x = self.head(x4)
        return {
                'middle1_fea': x1,
                'middle2_fea': x2,
                'middle3_fea': x3,
                'middle4_fea': x4,
                'output': x
                }
    
    def get_criterion(self):
        return Criterion()
    
    
class Criterion:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss() 
        
    def __call__(self, labels, outputs):        
        loss = self.criterion(outputs['output'], labels)
        return loss 