#%%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                Use this code to run inference on test data using IMGANet Model            #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Import the required libraries
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pickle
import random
import time
import sys
import csv

import PIL
from torch.utils.data import Dataset
import torchvision.transforms.functional as Ft

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batchsize = 1
num_classes = 2
att_type = 'two'
mask_guided = False
model_name = 'vgg19'

data_dir = "IMGA_test/vehicle_collision"

class Leafvein(Dataset):
    
    def __init__(self,
                 data_dir,
                 crop=None,
                 hflip=None,
                 vflip=None,
                 rotate=None,
                 erase=None,
                 mode='train'):
        """
        @ image_dir:   path to directory with images
        @ label_df:    df with image id (str) and label (0/1) - only for labeled test-set
        @ transforms:  image transformation; by default no transformation
        @ sample_n:    if not None, only use that many observations
        """
        self.data_dir = data_dir
        #self.transform = transform
        self.mode=mode
        #self.dataset=args.dataset
        with open(os.path.join(self.data_dir,'labels.pkl'),'rb') as df:
            self.label=pickle.load(df)
        self.img_files=os.listdir(os.path.join(self.data_dir, self.mode))
        self.crop=crop
        self.hflip=hflip
        self.vflip=vflip
        self.erase=erase
        self.rotate=rotate
        if mode=='train':
            self.transforms=transforms.Compose([
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            # the mean and std for leafvein dataset
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms=transforms.Compose([
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #transforms.Resize((720,1280)),
            transforms.ToTensor(),
            # the mean and std for leafvein dataset
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        
        print(f'Initialized datatset with {len(self.img_files)} images.\n')
    
#    @function_timer
    def _load_images(self):
        print('loading images in memory...')
        id2image = {}
        
        for file_name in self.img_files:
            img = PIL.Image.open(os.path.join(self.data_dir,self.mode, file_name))
            img = self.transforms(img)
            #id_ = file_name.split('.')[0]
            id_ = file_name
            id2image[id_] = img
        
        return id2image
    
    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        #id_=(file_name.split('.')[0])
        id_=(file_name)
        #img = PIL.Image.open(os.path.join(self.data_dir, self.dataset, self.mode,file_name)).convert('RGB')
        img = PIL.Image.open(os.path.join(self.data_dir,self.mode,file_name)).convert('RGB')
        #mask = PIL.Image.open(os.path.join(self.data_dir,'l2_mask',id_))
        #label = self.label[int(id_)]-1
        label = self.label[id_]
        #label = self.label[file_name]
        
        """
        if self.dataset=='btf':
            mask = PIL.Image.open(os.path.join(self.data_dir,self.dataset,'l2_mask',file_name)) # for btf dataset
            label=self.label[file_name]-1 # for btf dataset  
        else:
            mask = PIL.Image.open(os.path.join(self.data_dir,self.dataset,'l2_mask',id_+'.png')) # for soybean and hainan leaf dataset
            label=self.label[int(id_)]-1 # for soybean and hainan leaf dataset
        
        if self.hflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                
        if self.vflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)"""
        img = self.transforms(img)
        #mask = torch.FloatTensor(np.array(mask).astype(np.float))
        #mask = torch.FloatTensor(np.array(mask).astype(np.float64))
        
        """    
        if self.crop:
            # perform random crop
            h, w = img.shape[1], img.shape[2]
            pad_tb = max(0, self.crop[0] - h)
            pad_lr = max(0, self.crop[1] - w)
            img = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(img)
            h, w = img.shape[1], img.shape[2]
            i = random.randint(0, h - self.crop[0])
            j = random.randint(0, w - self.crop[1])
            img = img[:, i:i + self.crop[0], j:j + self.crop[1]]
            if self.mode=='train':
                mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 0)(mask)
                mask = mask[i:i + self.crop[0], j:j + self.crop[1]]
        if self.erase:
            if random.random() < 0.5:
                i, j, h, w, v=transforms.RandomErasing.get_params(img,scale=(0.02, 0.33), ratio=(0.3, 3.3))
                img = Ft.erase(img,i,j,h,w,v)
                mask = Ft.erase(mask.unsqueeze(0),i,j,h,w,v).squeeze(0)"""
        return img,label
    
    def __len__(self):
        return len(self.img_files)

# Data
print('==> Preparing data..')

#test = Leafvein(args,mode='test')
test = Leafvein(data_dir,mode='test')

testloader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=1)
#%%
class MGANET(nn.Module):
    def __init__(self,backbone_name,num_classes,att_type=None,mask_guided=False):
        super(MGANET, self).__init__()
        
        self.mask_guided=mask_guided
        if backbone_name=='densenet161':
            self.features=getattr(models,backbone_name)(pretrained=True).features
            self.classifier=nn.Linear(self.features[-1].num_features,num_classes)
        elif backbone_name=='efficientnet_v2_l':
            self.features=getattr(models,backbone_name)(pretrained=True).features
            #self.classifier=nn.Linear(self.features[-1].num_features,num_classes)
            self.classifier=nn.Linear(1280,num_classes)
        elif backbone_name[:6]=='resnet':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier=nn.Linear(512,num_classes)
        elif backbone_name=='mobilenet_v2':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier=nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.features.last_channel, num_classes)
            )
            self.features=self.features.features
        elif backbone_name=='vgg19':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            self.features=self.features.features
        else:
            self.features=getattr(models,backbone_name)(pretrained=True).features
            num_features=self.features[-1].num_features
            self.classifier=nn.Linear(num_features,num_classes)
            
        self.backbone_name=backbone_name
        
        self._freeze_layers(self.features)
        
        if backbone_name[:6]=='resnet':
            self.features=nn.Sequential(*list(self.features.children())[:-2])
        self.att_type=att_type
        '''
        if self.att_type=='three':
            self.classifier=nn.Linear(num_features*2,num_classes)
        '''
        kernel_size=1
        self.attention=nn.Conv2d(2,1,kernel_size=kernel_size, bias=False)

    def getAttFeats(self,att_map,features,type=None):
        # params: one for simple att*features
        # two for cat att*feat and features
        if type=='one':
            features=att_map*features
        elif type=='two':
            features=0.5*features+0.5*(att_map*features)
        elif type=='three':
            features=torch.cat((features,att_map*features),dim=1)
        else:
            pass
        return features
        
    def forward(self,x,mask=None):
        
        features = self.features(x)
        # output size 14*14
        #print(features.shape)
        #foreground attention
        fg_att=self.attention(torch.cat((torch.mean(features,dim=1).unsqueeze(1),\
                                                torch.max(features,dim=1)[0].unsqueeze(1)),dim=1))
        #fg_att=torch.flatten(torch.sigmoid(fg_att),1)
        fg_att=torch.sigmoid(fg_att)  
        features=self.getAttFeats(fg_att,features,type=self.att_type)
        if self.backbone_name=='densenet161':
            features = F.relu(features, inplace=True)
        if self.backbone_name=='efficientnet_v2_l':
            features = F.relu(features, inplace=True)
        if self.backbone_name=='vgg19':
            out=F.adaptive_avg_pool2d(features, (7, 7))
        else:
            out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        #### for calculating mse loss
        if self.mask_guided:
            h,w = fg_att.shape[2],fg_att.shape[3]
            mask=F.adaptive_avg_pool2d(mask, (h, w))
            fg_att = fg_att.view(fg_att.shape[0],-1)
            mask = mask.view(mask.shape[0],-1)
            
            mask += 1e-12
            max_elmts=torch.max(mask,dim=1)[0].unsqueeze(1)
            mask = mask/max_elmts
                                  
        return (out,fg_att,mask) if self.mask_guided else out
    
    def _freeze_layers(self,model):
        cnt,th=0,0
        print('freeze layers:')
        if self.backbone_name=='densenet161':
            th=9
        elif self.backbone_name=='efficientnet_v2_l':
            th=8
        elif self.backbone_name[:6]=='resnet':
            th=7
        elif self.backbone_name=='mobilenet_v2':
            th=11 # 10/19
        elif self.backbone_name=='vgg19':
            th=22 #9/16 conv
        else:
            th=10    
        for name, child in model.named_children():
            cnt+=1
            if cnt<th:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
                    print(name,name2,cnt)
                    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
   
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
#%%
# Model
print('==> Building model..')

model=MGANET(backbone_name=model_name,num_classes=num_classes,att_type=att_type,mask_guided=mask_guided)
#print(model)
net = model.to(device)
checkpoint_path = '/home/madhu-phd/1_Image_Classification/2_Class_Classification/Mask_Guided_Dataset/output_vgg19/checkpoint/mga_vgg19_lr1e4_bs_8.pth'
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])
# to print the model summary
#import torchvision
#from torchinfo import summary
#summary(model)
#summary(net))
#%%
test_acc=[]
test_ce_losses=[]
test_mse_losses=[]
output_pkldata = {}
y_gt = []

criterion = nn.CrossEntropyLoss()

net.eval()
test_loss = 0
ce_loss = 0
mse_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        ce_loss_ = criterion(outputs, targets)
        #mse_loss_=0
        loss =  ce_loss_ 
        
        ce_loss += ce_loss_.item()
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print(predicted, targets)
        key = testloader.dataset.img_files[batch_idx] #obtain the image name
        val = predicted.cpu().numpy()[0] #obtain the predicted class label for an image
        update_output = dict([(key,val)]) #form a dictionary of image_name and output prediction
        output_pkldata.update(update_output) 
        y_gt.append(targets.cpu().numpy()[0]) #obtain the ground truth for the image
        #inputs,targets,outputs=None,None,None

#save the output predictions of entire test set as a pkl file
with open('/home/madhu-phd/1_Image_Classification/2_Class_Classification/Mask_Guided_Dataset/output_vgg19/test_inference_lr1e4_bs_8.pkl','wb') as df:
    pickle.dump(output_pkldata,df)
    
y_true = torch.Tensor(y_gt) #convert all ground truths to a tensor
y_pred = torch.Tensor(list(output_pkldata.values())) #convert all predicted outputs to a tensor

#print('\nConfusion Matrix')
cm = confusion_matrix(y_true, y_pred)
#print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    
disp.plot()

#report = classification_report(y_true, y_pred,
#        target_names = ['No_Collision (Class 0)','Collision (Class 1)', 'Collided (Class 2)'])
report = classification_report(y_true, y_pred,
        target_names = ['No_Collision (Class 0)','Collision (Class 1)'])


print("Confusion Matrix")
print(disp.confusion_matrix)

print('\nClassification Report')
print(report)        
     
# Save checkpoint.
acc = 100.*correct/total

test_ce_losses.append(ce_loss/(batch_idx+1))
test_mse_losses.append(mse_loss/(batch_idx+1))
test_acc.append(acc)
#print('cur_acc:{0},best_acc:{1}:'.format(acc,best_acc))
print("finished")
print("Accuracy:{}".format(test_acc))
print("Cross-Entropy Loss:{}".format(test_ce_losses))
#print("Mean Square Error:{}".format(test_mse_losses))

inference_file = open('/home/madhu-phd/1_Image_Classification/2_Class_Classification/Mask_Guided_Dataset/output_vgg19/inference_summary.txt', 'w')
inference_file.write("\n Confusion Matrix:\n")
inference_file.write(str(cm))
inference_file.write("\nClassification Report:\n")
inference_file.write(report)
txt1 = "\n Accuracy for Test Data:{}".format(test_acc)
inference_file.write(txt1)
txt2 = "\n Cross Entropy Loss for Test Data:{}".format(test_ce_losses)
inference_file.write(txt2)
inference_file.close()
