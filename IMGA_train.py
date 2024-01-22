#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Use this code to train the IMGA network using the vehicular collision image classification dataset #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Import the required libraries
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

#%%
seed=213
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

test_acc=[]
train_acc=[]
train_ce_losses=[]
test_ce_losses=[]
train_mse_losses=[]
test_mse_losses=[]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = 0 # the epoch for the best accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

lr = 0.0001 # Set the learning rate
batchsize = 8 # Set the batch size for training
max_epoch = 500 # Set the number of epochs 
num_classes = 2 # Set to TWO - (No Collision & Collision)
att_type = 'two'
mask_guided = True
model_name = 'vgg19' # Feature extracting backbone network
#model_name = 'densenet161'
#model_name = 'efficientnet_v2_l'
#model_name = 'mobilenet_v2'

#%%
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

#%%
data_dir = "IMGA_train/vehicle_collision" # Specify the path of the training data

class VehicleCollision(Dataset):
    
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
        self.mode=mode
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
            transforms.ToTensor(),
            # the mean and std for VehicleCollision dataset
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms=transforms.Compose([
            transforms.ToTensor(),
            # the mean and std for VehicleCollision dataset
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
            id_ = file_name
            id2image[id_] = img
        
        return id2image
    
    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        id_=(file_name)
        img = PIL.Image.open(os.path.join(self.data_dir,self.mode,file_name)).convert('RGB')
        mask = PIL.Image.open(os.path.join(self.data_dir,'l2_mask',id_))
        label = self.label[id_]
        
        if self.hflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                
        if self.vflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        img = self.transforms(img)
        mask = torch.FloatTensor(np.array(mask).astype(np.float64))
            
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
                mask = Ft.erase(mask.unsqueeze(0),i,j,h,w,v).squeeze(0)
        return img,label,mask
    
    def __len__(self):
        return len(self.img_files)

#%%
# Data
print('==> Preparing data..')

train = VehicleCollision(data_dir,crop=False,hflip=False,vflip=False,erase=False,mode='train')
test = VehicleCollision(data_dir,mode='test')

trainloader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=1)
testloader = DataLoader(test, batch_size=8, shuffle=False, num_workers=1)

#%%
class IMGANET(nn.Module):
    def __init__(self,backbone_name,num_classes,att_type=None,mask_guided=False):
        super(IMGANET, self).__init__()
        
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
        
        # for calculating mse loss
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
            th=8
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

model=IMGANET(backbone_name=model_name,num_classes=num_classes,att_type=att_type,mask_guided=mask_guided)
#print(model)

net = model.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

#%%
filename = "IMGA/output/imga_vgg19_lr1e4_bs_8.csv"  #Give the file name with path for saving the output csv file

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    mse_loss = 0
    ce_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, masks) in enumerate(trainloader):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad()
        if mask_guided:
            outputs,fg_atts,masks = net(inputs,masks)
            mse_loss_ = mse(fg_atts,masks)
            ce_loss_ = criterion(outputs, targets)
            loss =  ce_loss_ + 0.1 * mse_loss_
            mse_loss += mse_loss_.item()
        else:
            outputs = net(inputs)
            ce_loss_ = criterion(outputs, targets)
            mse_loss_=0
            loss =  ce_loss_
            mse_loss += 0
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        ce_loss += ce_loss_.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'CE: %.3f, MSE: %.5f | Acc: %.3f%% (%d/%d)'
                     % (ce_loss/(batch_idx+1), mse_loss/(batch_idx+1) ,100.*correct/total, correct, total))
    train_ce_losses.append(ce_loss/(batch_idx+1))
    train_mse_losses.append(mse_loss/(batch_idx+1))
    train_acc.append(100.*correct/total)

def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    ce_loss = 0
    mse_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,masks) in enumerate(testloader):
            inputs, targets,masks= inputs.to(device), targets.to(device),masks.to(device)
            if mask_guided:
                outputs,fg_atts,masks = net(inputs,masks)
                mse_loss_ = mse(fg_atts,masks)
                ce_loss_ = criterion(outputs, targets)
                loss =  ce_loss_ + 0.1 * mse_loss_
                mse_loss += mse_loss_.item()
            else:
                outputs = net(inputs)
                ce_loss_ = criterion(outputs, targets)
                mse_loss_=0
                loss =  ce_loss_
                mse_loss += 0
            ce_loss += ce_loss_.item()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
            progress_bar(batch_idx, len(testloader), 'CE: %.3f, MSE: %.5f | Acc: %.3f%% (%d/%d)'
                         % (ce_loss/(batch_idx+1), mse_loss/(batch_idx+1), 100.*correct/total, correct, total))
            inputs,targets,outputs=None,None,None

    # Save checkpoint. Save the model parameters that has best validation accuracy.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir("IMGA/output/checkpoint"): # Give the path of the checkpoint folder to save the model
            os.mkdir("IMGA/output/checkpoint")
        torch.save(state, "IMGA/output/checkpoint/imga_vgg19_lr1e4_bs_8.pth") # Give the file name with path for the model to be saved
        best_acc = acc
        best_epoch=epoch
    test_ce_losses.append(ce_loss/(batch_idx+1))
    test_mse_losses.append(mse_loss/(batch_idx+1))
    test_acc.append(acc)
    print('cur_acc:{0},best_acc:{1}:'.format(acc,best_acc))

    with open(filename,'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',lineterminator='\n')
        csvwriter.writerow([[train_ce_losses[-1],train_mse_losses[-1],train_acc[-1],test_ce_losses[-1],test_mse_losses[-1],test_acc[-1],scheduler.get_lr()]])

#%%
for epoch in range(start_epoch, start_epoch+max_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
    #learningrate = scheduler.get_last_lr()
    print(scheduler.get_lr())
#%%
