#The following file paths are all absolute paths. You can replace them with relative paths at runtime, and the files are located in their respective folders.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import sys
import os
import csv
from tqdm import tqdm, trange
sys.path.append("utility_LSPN/")
from tensorboardX import SummaryWriter
from scipy.integrate import odeint
from Utility import data_collecter
from LSPN_test import LSPN_Mamba
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

def K_loss(data,net):
    train_traj_num,steps,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.FloatTensor(data).to(device)
    X_pred = net.forward(data[:,:steps-1,:])#.transpose(0, 1)
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        X_current = X_pred[:,i,:]
        Y = data[:,i+1,:]
        Err = X_current-Y
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err),axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.min(torch.mean(torch.abs(Err),axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list),np.array(mean_loss_list)


#loss function
def Klinear_loss(data,net,mse_loss,gamma=0.99):
    train_traj_num, steps,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.FloatTensor(data).to(device)
    # print(data[:steps-1,:,:].transpose(0, 1).shape)
    X_pred = net.forward(data[:,:steps-1,:])#.transpose(0, 1)) #.forward
    X_pred = X_pred#.transpose(0, 1)
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = X_pred[:,i,:]
        Y = data[:,i+1,:]
        beta_sum += beta
        loss += beta*mse_loss(X_current,Y) #+ mse_loss(U_pred[i,:,:],data[i,:,:u_dim])
        beta *= gamma
    loss = loss/beta_sum
    return loss


def read_rossler_dataset_original_shape(file_path):
    data = np.load(file_path)
    return data


def train(env_name,train_steps = 100000,suffix="",Ktrain_samples=20000):   
    Ktrain_samples = Ktrain_samples
    Ktest_samples = 10000#20000
    Ksteps = 50
    Kbatch_size = 100
    gamma = 0.8
    X_original_shape = read_rossler_dataset_original_shape('utility_LSPN/henon_map_data_filtered.npy')
    Ktrain_data = X_original_shape[:Ktrain_samples, :Ksteps, :]
    print("train data ok!{}".format(Ktrain_data.shape))
    Ktest_data = X_original_shape[Ktrain_samples:Ktrain_samples+Ktest_samples, :Ksteps, :]
    print("test data ok!{}".format(Ktest_data.shape))
    net = LSPN_Mamba(
        d_model=2, # Model dimension d_model
        d_state=8,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=4,    # Block expansion factor
    ).to("cuda")
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.float()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    eval_step = 100
    best_loss = 1000.0
    best_state_dict = {}
    logdir = "DATA/Mamba_data_raw/"+suffix+"/KNonlinearmamba_"+"henon_map"+"samples{}".format(Ktrain_samples)
    if not os.path.exists( "DATA/Mamba_data_raw/"+suffix):
        os.makedirs( "DATA/Mamba_data_raw/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in trange(train_steps, desc="henon_map", unit="epoches"):
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[Kindex[:Kbatch_size],:,:]
        Kloss = Klinear_loss(X,net,mse_loss,gamma)
        optimizer.zero_grad()
        Kloss.backward()
        optimizer.step() 
        writer.add_scalar('Train/loss',Kloss,i)
        if (i+1) % eval_step ==0:
            with torch.no_grad():
                Ktrainloss = Kloss
                Kloss = Klinear_loss(Ktest_data,net,mse_loss,gamma)
                writer.add_scalar('Eval/loss',Kloss,i)
                writer.add_scalar('Eval/best_loss',best_loss,i)
                if Kloss<best_loss:
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict}
                    torch.save(Saved_dict,logdir+".pth")
                print("Step:{} Ktrainloss:{} Eval K-loss:{} ".format(i+1,Ktrainloss.detach().cpu().numpy(),Kloss.detach().cpu().numpy()))
        writer.add_scalar('Eval/best_loss',best_loss,i)
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix)
    pass

if __name__ == "__main__":
    env_names = ["henon_map"]#"MountainCarContinuous-v0","MountainCarContinuous-v0",,"CartPole-v1","MountainCarContinuous-v0","Pendulum-v1"
    for i in env_names:
        parser = argparse.ArgumentParser()
        parser.add_argument("--env",type=str,default=i)
        parser.add_argument("--suffix",type=str,default="mamba_testhenon_map_100k")
        args = parser.parse_args()
        main()
    pass
        

