
import timm 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms
from .model import BaseModel

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )    

class BeYourOwnTeacher(BaseModel):
    def __init__(self, model_name, pretrained = True, num_class = 15, 
                    temperature = 3, alpha:float = 0.1, beta:float = 1e-6):
        super(BeYourOwnTeacher,self).__init__(
            model_name = model_name,
            pretrained = pretrained, 
            num_class = num_class 
        )
        self.criterion = Criterion(temperature, alpha, beta)
        self.expansion     = list(self.block1.children())[0].expansion
        
        self.downsample1_1 = nn.Sequential(
                                            conv1x1(64 * self.expansion, 512 * self.expansion, stride=8),
                                            nn.BatchNorm2d(512 * self.expansion),
        )
        self.bottleneck1_1 = branchBottleNeck(64 * self.expansion, 512 * self.expansion, kernel_size=8)
        self.avgpool1      = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1    = nn.Linear(512 * self.expansion, num_class)


        self.downsample2_1 = nn.Sequential(
                                            conv1x1(128 * self.expansion, 512 * self.expansion, stride=4),
                                            nn.BatchNorm2d(512 * self.expansion),
            )
        self.bottleneck2_1 = branchBottleNeck(128 * self.expansion, 512 * self.expansion, kernel_size=4)
        self.avgpool2      = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2    = nn.Linear(512 * self.expansion, num_class)


        self.downsample3_1 = nn.Sequential(
                                            conv1x1(256 * self.expansion, 512 * self.expansion, stride=2),
                                            nn.BatchNorm2d(512 * self.expansion),
        )
        self.bottleneck3_1 = branchBottleNeck(256 * self.expansion, 512 * self.expansion, kernel_size=2)
        self.avgpool3      = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3    = nn.Linear(512 * self.expansion, num_class)
        
        self.avgpool       = list(self.model.children())[8]
        self.fc            = list(self.model.children())[9]        
        
    def forward(self,x):
        x = self.init_layer(x)
        
        x = self.block1(x)
        x1 = x.detach()
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc2(middle_output1)
        
        x = self.block2(x)
        x2 = x 
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)
        
        x = self.block3(x)
        x3 = x.detach()
        middle_output3 = self.bottleneck3_1(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)
        
        x = self.block4(x)
        x4 = x.detach()
        x = self.avgpool(x)
        middle4_fea = x 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
#        return [x, 
#                middle_output1, middle_output2, middle_output3, 
#                final_fea, middle1_fea, middle2_fea, middle3_fea]
        
        return  {
                'middle1_fea': middle1_fea,
                'middle2_fea': middle2_fea,
                'middle3_fea': middle3_fea,
                'middle4_fea': middle4_fea,
                'middle_output1' : middle_output1,
                'middle_output2' : middle_output2,
                'middle_output3' : middle_output3,
                'output': x,
                'middle_raw_fea': [x1,x2,x3,x4]
                }
    
    def get_criterion(self):
        return self.criterion
        
    
    
class Criterion:
    def __init__(self, temperature:int = 3, alpha:float = 0.1, beta:float = 1e-6):
        self.temperature = temperature
        self.clf_criterion = nn.CrossEntropyLoss()
        self.alpha = alpha 
        self.beta  = beta 
        
    def feature_criterion(self,fea, target_fea):
        loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).sum()
    
        
    def kd_criterion(self,output, target_output,temperature):
        output = output / temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd
    
    def __call__(self, target, preds: list):
        middle1_fea    = preds['middle1_fea'] 
        middle2_fea    = preds['middle2_fea'] 
        middle3_fea    = preds['middle3_fea'] 
        middle4_fea    = preds['middle4_fea'] 
        middle_output1 = preds['middle_output1'] 
        middle_output2 = preds['middle_output2'] 
        middle_output3 = preds['middle_output3'] 
        output         = preds['output'] 
        
        # Classification Loss 
        clf_loss     = self.clf_criterion(output, target)
        middle1_loss = self.clf_criterion(middle_output1, target)
        middle2_loss = self.clf_criterion(middle_output2, target)
        middle3_loss = self.clf_criterion(middle_output3, target)
        del target
        
        # Knowledge Distillation Loss 
        teacher = output / self.temperature
        teacher = torch.softmax(teacher, dim=1)
        del output 
        
        kd1_loss = self.kd_criterion(middle_output1, teacher, self.temperature) * (self.temperature**2)
        kd2_loss = self.kd_criterion(middle_output2, teacher, self.temperature) * (self.temperature**2)
        kd3_loss = self.kd_criterion(middle_output3, teacher, self.temperature) * (self.temperature**2)
        del teacher, middle_output1, middle_output2, middle_output3 
        
        # Feature distance Loss 
        feature1_loss = self.feature_criterion(middle1_fea, middle4_fea.detach())
        feature2_loss = self.feature_criterion(middle2_fea, middle4_fea.detach())
        feature3_loss = self.feature_criterion(middle3_fea, middle4_fea.detach())
        
        # Total Loss 
        total_loss = (1 - self.alpha) * (clf_loss + middle1_loss + middle2_loss + middle3_loss) + \
            self.alpha * (kd1_loss + kd2_loss + kd3_loss) + \
                self.beta * (feature1_loss + feature2_loss + feature3_loss)
            
        return total_loss 
        