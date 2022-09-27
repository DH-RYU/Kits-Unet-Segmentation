from kits_model import *
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support

from starter_code.utils import load_case
from starter_code.visualize import visualize
from starter_code.evaluation import evaluate,small_evaluate
from torch.utils.tensorboard import SummaryWriter


class Solver(object):
    def __init__(self,config):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.num_layers = config.num_layers
        self.residual_path = config.residual_path
        self.resize = config.resize
        self.random_crop = config.random_crop

        self.train_load = config.train_load
        self.start_epoch = config.start_epoch

        self.lr = config.lr
        self.n_epochs = config.n_epochs
        self.device = config.device
        self.eps = config.eps
        self.batch_size = config.batch_size
        
        self.mode = config.mode
        self.load_epoch = config.load_epoch
        self.kits_directory = config.kits_directory

        self.save_path = config.save_path
        self.save_name = config.save_name
    def load_data(self):
        self.train_dataset = Kits_Dataset(self.kits_directory,mode='train',resize=self.resize,random_crop=self.random_crop)
        self.val_dataset = Kits_Dataset(self.kits_directory,mode='val',resize=self.resize,random_crop=False)
        self.test_dataset = Kits_Dataset(self.kits_directory,mode='test',resize=self.resize,random_crop=False)
        
        self.train_loader = DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size = 1,shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,batch_size = 1,shuffle=False)
        self.val_plot_loader = DataLoader(self.val_dataset,batch_size = 1 ,shuffle=True)
     
    def build_model(self):
        self.load_data()
        self.unet = U_Net(self.input_dim,self.output_dim,self.num_layers,self.residual_path).to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(),lr = self.lr,eps = self.eps)
        weight = torch.Tensor([0.2,0.3,0.5]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        if self.train_load == True :
            checkpoint = torch.load(os.path.join(self.save_path,'saved_'+self.save_name+'_'+str(self.start_epoch)+'.pth'))
            self.unet.load_state_dict(checkpoint['unet'])
    def save_model(self,epoch):
        os.makedirs(self.save_path,exist_ok=True)
        torch.save({'unet' : self.unet.state_dict(),
                    'epoch' : epoch
                    },os.path.join(self.save_path,'saved_'+self.save_name+'_'+str(epoch)+'.pth'))

        print('Model Saved!')
    def normalize(self,x,downer,upper):
        x[x<downer] = downer
        x[x>upper] = upper
        x = (x - downer)/(upper-downer)
        return x
    def score(self,target,prediction):
        return precision_recall_fscore_support(target,prediction)
    def save_log(self,epoch,total_train_loss,total_val_loss,tk_dice,tu_dice):
        data = iter(self.val_plot_loader)
        while True :
            X,y = next(data)
            if 2 in y :
                break
        X = X.to(self.device)
        y = y.to(self.device)
        batch_size = X.shape[0]
        self.unet.eval()
        prediction = self.unet(X)
        ## Coloring on original image
        prediction = torch.round(torch.softmax(prediction,dim=1)).argmax(dim=1).unsqueeze(1)
        X_max = torch.max(X)
        X_min = torch.min(X)
        y = y
        y = torch.cat([y]*3,dim=1)
        prediction = torch.cat([prediction]*3,dim=1)
        label = (torch.cat([X]*3,dim=1)-X_min)/(X_max-X_min)
        pred = (torch.cat([X]*3,dim=1)-X_min)/(X_max-X_min)
        ones = y==1
        ones[:,[0,2]] = False
        twos = y==2
        twos[:,[1,2]] = False
        label[ones] += 0.2
        label[twos] += 0.2
        ones = prediction==1
        ones[:,[0,2]] = False
        twos = prediction==2
        twos[:,[1,2]] = False
        pred[ones] += 0.2
        pred[twos] += 0.2
        ## Coloring on original image

        grid = torch.cat([X.repeat(1,3,1,1),label,pred],dim=0)
        grid = torchvision.utils.make_grid(grid,nrow=int(batch_size))

        self.writer.add_image('Image/Kits', grid, epoch)
        self.writer.add_scalars("Loss", {'train_loss' : total_train_loss/len(self.train_loader),'validation_loss' : total_val_loss/len(self.val_loader)}, epoch)
        self.writer.add_scalar("Measurement/tk_dice",tk_dice,epoch)
        self.writer.add_scalar("Measurement/tu_dice",tu_dice,epoch)
    def train(self):
        self.writer = SummaryWriter('./runs/'+self.save_name)
        best_loss = 100 
        best_dice = 0
        self.unet.train()
        for epoch in tqdm(range(self.start_epoch,self.n_epochs)):
            total_train_loss = 0
            tk_dice,tu_dice = 0,0
            tk_count = 0.000001
            tu_count = 0.000001
            for i,(X,y) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                X = X.to(self.device)
                y = y.to(self.device)
                if torch.sum(y) == 0 :
                    if random.random() > 0.2 :
                        continue
                pred = self.unet.forward(X)
                pred = pred.view(X.shape[0],3,-1)
                pred_idx = torch.softmax(pred,dim=1).argmax(dim=1)
                y = y.view(X.shape[0],-1).long()
                loss = self.criterion(pred,y) # Cross entropy loss
                total_train_loss += loss.item()
                if (2 in y) or (2 in pred_idx) :
                    tk,tu = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                    tk_dice += tk
                    tu_dice += tu
                    tk_count += 1
                    tu_count += 1
                elif ((1 in y) and (not(2 in y))) or ((1 in pred_idx) and (not(2 in pred_idx))) :
                    tk,_ = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                    tk_dice += tk
                    tk_count += 1
                loss.backward()
                self.optimizer.step()
                if i % 1000 == 0 :
                    print("\n")
                    print(epoch,'epochs ',i,'iteration/',len(self.train_loader),':')
                    print(tk_dice/tk_count,tu_dice/tu_count)
                    tk_dice,tu_dice = 0,0
                    tk_count = 0.000001
                    tu_count = 0.000001
                break
            total_val_loss = 0
            tk_dice = 0
            tu_dice = 0
            tk_count = 0.000001
            tu_count = 0.000001
            self.unet.eval()
            with torch.no_grad():
                for i,(X,y) in tqdm(enumerate(self.val_loader)):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    pred = self.unet.forward(X)
                    pred = pred.view(X.shape[0],3,-1)
                    pred_idx = torch.softmax(pred,dim=1).argmax(dim=1)
                    y = y.view(X.shape[0],-1).long()
                    loss = self.criterion(pred,y) # Cross entropy loss
                    total_val_loss += loss.item()
                    if (2 in y) or (2 in pred_idx) :
                        tk,tu = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                        tk_dice += tk
                        tu_dice += tu
                        tk_count += 1
                        tu_count += 1
                    elif ((1 in y) and (not(2 in y))) or ((1 in pred_idx) and (not(2 in pred_idx))) :
                        tk,_ = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                        tk_dice += tk
                        tk_count += 1
                    self.save_log(epoch,total_train_loss,total_val_loss,tk_dice/tk_count,tu_dice/tu_count)
                print(epoch, "epoch")
                print(tk_dice/tk_count,tu_dice/tu_count)
                ## Tensorboard Log Saving ##
                ## Tensorboard Log Saving ##
                self.save_model(epoch)
    def test(self):
        checkpoint = torch.load(os.path.join(self.save_path,'saved_'+self.save_name+'_'+str(self.load_epoch)+'.pth'))
        self.unet.load_state_dict(checkpoint['unet'])
        total_test_loss = 0
        tk_dice = 0
        tu_dice = 0
        tk_count = 0.000001
        tu_count = 0.000001
        self.unet.eval()
        with torch.no_grad():
            for i,(X,y) in tqdm(enumerate(self.test_loader)):
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.unet.forward(X)
                pred = pred.view(X.shape[0],3,-1)
                pred_idx = torch.softmax(pred,dim=1).argmax(dim=1)
                y = y.view(X.shape[0],-1).long()
                loss = self.criterion(pred,y) # Cross entropy loss
                total_test_loss += loss.item()
                if (2 in y) or (2 in pred_idx) :
                    tk,tu = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                    tk_dice += tk
                    tu_dice += tu
                    tk_count += 1
                    tu_count += 1
                elif ((1 in y) and (not(2 in y))) or ((1 in pred_idx) and (not(2 in pred_idx))) :
                    tk,_ = small_evaluate(y.cpu().detach().numpy(),pred_idx.cpu().detach().numpy())
                    tk_dice += tk
                    tk_count += 1
            print(tk_dice/tk_count,tu_dice/tu_count)

