# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import copy

from sklearn.model_selection import train_test_split

# %%
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using : ",device)

# %%
transform_train=transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0,translate=(0.1,0.1)),
    transforms.ToTensor()
])

transform_test=transforms.Compose([
    transforms.ToTensor()
])

trainset=torchvision.datasets.CIFAR100(root="./data",train=True,download=True,transform=transform_train)
testset=torchvision.datasets.CIFAR100(root="./data",train=False,download=True,transform=transform_test)

train_idx,val_idx=train_test_split(
    np.arange(len(trainset)),test_size=0.1,stratify=trainset.targets,random_state=24
)

train_subset=Subset(trainset,train_idx)
val_subset=Subset(trainset,val_idx)


batch_size=128
trainloader=DataLoader(train_subset,batch_size=batch_size,shuffle=True,num_workers=2)
valloader=DataLoader(val_subset,batch_size=batch_size,shuffle=True,num_workers=2)
testloader=DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

# %%
class CustomCNN(nn.Module):
    def __init__(self,num_classes=100):
        super().__init__()
        self.features=nn.Sequential(

            #Block 1
            nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            #Block 2
            nn.Conv2d(32,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            #Block 3
            nn.Conv2d(64,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x


model=CustomCNN(num_classes=100).to(device)
print(model)

# %%
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

schedular=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3)

# %%
epochs=60
patience=8
best_model_wts=copy.deepcopy(model.state_dict())
best_val_loss=float("inf")
early_stop_counter=0

train_losss,val_losss,train_accs,val_accs=[],[],[],[]

for epoch in range(1,epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")

    model.train()
    running_loss,correct,total=0.0,0,0
    for imgs,labels in trainloader:
        imgs,labels=imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(imgs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item() * imgs.size(0)
        _,preds=outputs.max(1)
        correct+=preds.eq(labels).sum().item()
        total+=labels.size(0)
    
    trian_loss=running_loss/total
    train_acc=correct/total
    train_losss.append(trian_loss)
    train_accs.append(train_acc)


    model.eval()
    val_loss,val_correct,val_total=0.0,0,0
    with torch.no_grad():
        for imgs,labels in valloader:
            imgs,labels=imgs.to(device),labels.to(device)
            outputs=model(imgs)
            loss=criterion(outputs,labels)
            val_loss+=loss.item() * imgs.size(0)
            _,preds=outputs.max(1)
            val_correct+=preds.eq(labels).sum().item()
            val_total+=labels.size(0)
        
    val_loss/=val_total
    val_acc=val_correct/val_total
    val_losss.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train loss {trian_loss:.4f} acc {train_acc:.3f} | Val loss {val_loss:.4f} acc {val_acc:.3f}")

    schedular.step(val_loss)


    if val_loss < best_val_loss:
        best_val_loss=val_loss
        best_model_wts=copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(),"best_custom_cnn.pth")
        print("Saved best Model")
        early_stop_counter=0
    else :
        early_stop_counter+=1
        if early_stop_counter>=patience:
            print("Early Stopping")
            break

model.load_state_dict(best_model_wts)

# %%



