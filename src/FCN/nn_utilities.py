import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from collections import Counter
import os
# import time
# import wandb
import datetime
import argparse
# wandb.init(project="DT_ML")

def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does Linear_Model decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, isTerminal, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.isTerminal = isTerminal

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.isTerminal:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# # either initialize early stopping or learning rate scheduler
# if args['lr_scheduler']:
#     print('INFO: Initializing learning rate scheduler')
#     lr_scheduler = LRScheduler(optimizer)
#     # change the accuracy, loss plot names and model name
#     loss_plot_name = 'lrs_loss'
#     acc_plot_name = 'lrs_accuracy'
#     model_name = 'lrs_model'
# if args['early_stopping']:
#     print('INFO: Initializing early stopping')
#     early_stopping = EarlyStopping()
#     # change the accuracy, loss plot names and model name
#     loss_plot_name = 'es_loss'
#     acc_plot_name = 'es_accuracy'
#     model_name = 'es_model'

class data_load_processing():
    def __init__(self, folder_add, usingDTdata):
        self.realdata = True
        self.batch_size = 2000
        self.isavg = False
        self.onetile= False
        self.ratio = 0.8
        self.folder = folder_add
        self.usingDTdata = usingDTdata
        # self.usingEncode = usingEncode
        self.BSLoc = np.asarray([358448, 173524]).reshape(1,2)

    # 1 load all initial data
    def data_load(self):
        df = pd.read_csv(self.folder)
        x_min = df.xmin.to_numpy()
        y_min = df.ymin.to_numpy()
        rsrp = df.rsrp.to_numpy()
        nodes = df[df.columns[-1]].to_numpy()
        # xy_all = np.vstack((x_min, y_min, rsrp, nodes)).T
        x = np.vstack((x_min, y_min)).T
        y= np.vstack((rsrp, nodes)).T

        self.x = x
        self.y = y

        # np.linalg.norm(x, self.BSLoc)
        # return x,y
        # np.sqrt(np.sum(np.square(x[11000] - self.BSLoc)))
    def DT_data_load(self):
        self.DT_label = scio.loadmat('DT_training_input.mat')['DT_training_input']
        self.DT_train = scio.loadmat('DT_training_label.mat')['DT_training_label']


    # 2 remove outlier
    def remove_outlier_Hampel(self, x):
        med=np.median(x)
        List=abs(x-med)
        cond=np.median(List)*3.5
        # good_list=List[~(List>cond)]
        # good_x = good_list + med
        good_index = np.where((~(List>cond))==True)
        good_index = np.asarray(good_index).T.flatten()
        return good_index

    # 3 normalisation
    def data_normalisation(self, x, y):
        # remove outlier by using Hampel function
        x = x.astype(float)
        good_index_x = self.remove_outlier_Hampel(x[:,0])
        mid_x = x[good_index_x, :]
        mid_y = y[good_index_x, :]

        good_index_y = self.remove_outlier_Hampel(mid_x[:,1])
        mid_x_final = mid_x[good_index_y, :]
        mid_y_final = mid_y[good_index_y, :]

        x = mid_x_final
        y = mid_y_final

        for i in range(len(x[0])):
            x[:, i] = (x[:, i] - min(x[:, i])) / (max(x[:, i]) - min(x[:, i]))

        self.nor_x = x
        self.nor_y = y

        # return x, y

    # 4 data sampling
    def data_sampling(self, x, y):
        x_min = x[:, 0]
        y_min = x[:, 1]
        xy_all = np.hstack((x, y))
        a = np.unique(x_min)
        b = np.unique(y_min)

        xy_yscreened = [[np.zeros((500, 4)) for j in range(len(b))] for i in range(len(a))]
        xy_loc = [[np.zeros((1,2)) for j in range(len(b))] for i in range(len(a))]

        for j in range(len(b)):
            for i in range(len(a)):
                x_index = np.asarray(np.where(xy_all[:, 0] == a[i])).flatten()
                xy_xscreened = xy_all[x_index, :]

                y_index = np.asarray(np.where(xy_xscreened[:, 1] == b[j])).flatten()
                if y_index.size == 0:
                    xy_yscreened[i][j] = []
                else:
                    xy_yscreened[i][j] = xy_xscreened[y_index, :]
                xy_loc[i][j] = np.hstack((a[i],b[j]))

        xy_reorganise = xy_yscreened
        t = len(xy_reorganise)
        n = len(xy_reorganise[0])

        mu_tmp = []
        sigma_tmp = []
        final_loc= []
        ttp = np.empty(0)

        for j in range(n):
            for i in range(t):
                if len(xy_reorganise[i][j]):
                    tile_mu, tile_sigma = self.tile_dis(xy_reorganise[i][j])
                    if self.isavg:
                        mu_tmp.append(tile_mu)
                        sigma_tmp.append(tile_sigma)
                        final_loc.append(xy_loc[i][j])
                    else:
                        tile_array = np.asarray(tile_mu)
                        ttp = np.hstack((ttp, tile_array))
                        for ii in range(len(tile_array)):
                            sigma_tmp.append(tile_sigma)
                            final_loc.append(xy_loc[i][j])
        if self.isavg:
            label = np.vstack((np.asarray(mu_tmp),np.asarray(sigma_tmp))).T
            num = np.asarray(mu_tmp)
            # label = num.reshape(len(num), 1)
            train = np.asarray(final_loc)
        else:
            # label = ttp.reshape(len(ttp),1)
            label = np.hstack((ttp.reshape(len(ttp),1), np.asarray(sigma_tmp).reshape(len(sigma_tmp), 1)))
            np.asarray(sigma_tmp)
            train = np.asarray(final_loc)

        # self.distance = np.linalg.norm(train - self.BSLoc, axis=1).reshape(-1, 1)
        # train = np.hstack((train, self.distance))
        #
        # for i in range(len(train[0])):
        #     train[:, i] = (train[:, i] - min(train[:, i])) / (max(train[:, i]) - min(train[:, i]))

        self.train = train
        self.label = label

        return train, label



    # 5 split it into training and test set
    def data_splitting(self, x, y):
        """
        :param x: training features
        :param y: training labels
        :return:
        """

        data_len = x.shape[0]
        indices = np.random.permutation(data_len)
        training_idx, test_idx = indices[:int(data_len * self.ratio)], indices[int(data_len * self.ratio):]
        x_train, x_test = x[training_idx], x[test_idx, :]
        y_train, y_test = y[training_idx], y[test_idx, :]
        train_data = np.hstack((x_train, y_train))


        test_data = np.hstack((x_test, y_test))

        # c, ia, ic = np.unique(train_data[:,0:2], return_index=True, return_inverse=True, axis=0)
        #
        # c1, ia1, ic1 = np.unique(ic, return_index=True, return_inverse=True, axis=0)
        #
        #
        # sorted_data = list(Counter(ic).items())

        # for t in range(len(sorted_data)):
        #     if sorted_data[t][1] == 1:
        #         pass





        print(f'The number of training samples is {len(train_data)}')
        print(f'The number of test samples is {len(test_data)}')

        if self.onetile:
            print(f'The mean of samples is {np.mean(train_data[:, 2])}')
            print(f'The variance of samples is {np.var(train_data[:, 2], ddof=1)}')
            train_data[:, 0] = train_data[0, 0]
            train_data[:, 1] = train_data[0, 1]
        return train_data, test_data


    def tile_dis(self, tmp): # returns the mean and variance of RSRP for a given tile.
        newdata = []
        for k in range(len(tmp)):
            for i in range(tmp[k, -1].astype(int)):
                newdata.append(tmp[k, -2])
        if self.isavg == True:
            tile_mu = np.mean(np.asarray(newdata))
        else:
            tile_mu = np.asarray(newdata)
        tile_sigma=np.var(np.asarray(newdata), ddof=1)

        if np.isnan(tile_sigma):
            tile_sigma = 1   # assumption of ti;e
        tile_all = np.asarray(newdata)
        return tile_mu, tile_sigma

    def Data_preprocessing(self):

        if self.usingDTdata:
            self.DT_data_load()
            self.data_normalisation(self.DT_train, self.DT_label)
            train_data, test_data = self.data_splitting(self.nor_x, self.nor_y)
        else:
            self.data_load()
            self.data_normalisation(self.x, self.y)
            self.data_sampling(self.nor_x, self.nor_y)
            train_data, test_data = self.data_splitting(self.train, self.label)

        train_dataset = GetLoader(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        test_dataset = GetLoader(test_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader, test_loader, train_dataset, test_dataset
        # return tile_mu


class encoded_data_processing(data_load_processing):
    def __init__(self, folder_add, usingDTdata, xynodes_all, encode_out):
        super(encoded_data_processing, self).__init__(folder_add, usingDTdata)
        self.xynodes_all = xynodes_all
        self.encode_out = encode_out

    def encode_data_load(self):
        xynodes_all = self.xynodes_all # 4 column
        encode_out = self.encode_out  # 3 column

        # xynodes_all = scio.loadmat('tmp1.mat')['tmp1']  # 4 column
        # encode_out = np.load("encode_out.npy")  # 3 column
        # for i in range(len(encode_out[0])):
        # i = 0
            # encode_out[:, i] = (encode_out[:, i] - min(encode_out[:, i])) / (max(encode_out[:, i]) - min(encode_out[:, i]))
        # encode_out = np.delete(encode_out, 0, axis=1)
        self.encode_alldata = np.hstack((xynodes_all[:, 0:2], encode_out[:, :], xynodes_all[:, 2:5]))
        self.num_colums = np.size(self.encode_alldata, 1)

    def data_sampling(self, x, y):
        x_min = x[:, 0]
        y_min = x[:, 1]
        xy_all = np.hstack((x, y))
        a = np.unique(x_min)
        b = np.unique(y_min)

        xy_yscreened = [[np.zeros((500, self.num_colums)) for j in range(len(b))] for i in range(len(a))]
        xy_loc = [[np.zeros((1, 2)) for j in range(len(b))] for i in range(len(a))]

        for j in range(len(b)):
            for i in range(len(a)):
                x_index = np.asarray(np.where(xy_all[:, 0] == a[i])).flatten()
                xy_xscreened = xy_all[x_index, :]

                y_index = np.asarray(np.where(xy_xscreened[:, 1] == b[j])).flatten()
                if y_index.size == 0:
                    xy_yscreened[i][j] = []
                else:
                    xy_yscreened[i][j] = xy_xscreened[y_index, :]
                xy_loc[i][j] = np.hstack((a[i], b[j]))

        xy_reorganise = xy_yscreened
        t = len(xy_reorganise)
        n = len(xy_reorganise[0])

        mu_tmp = []
        sigma_tmp = []
        final_loc = []
        encode_tmp = []
        ttp = np.empty(0)

        for j in range(n):
            for i in range(t):
                if len(xy_reorganise[i][j]):
                    tile_mu, tile_sigma, tile_encode = self.tile_dis(xy_reorganise[i][j])
                    if self.isavg:
                        mu_tmp.append(tile_mu)
                        sigma_tmp.append(tile_sigma)
                        final_loc.append(xy_loc[i][j])
                    else:
                        tile_array = np.asarray(tile_mu)
                        ttp = np.hstack((ttp, tile_array))
                        for ii in range(len(tile_array)):
                            sigma_tmp.append(tile_sigma)
                            final_loc.append(xy_loc[i][j])
                            encode_tmp.append(tile_encode)
        if self.isavg:
            label = np.vstack((np.asarray(mu_tmp), np.asarray(sigma_tmp))).T
            num = np.asarray(mu_tmp)
            # label = num.reshape(len(num), 1)
            train = np.asarray(final_loc)
        else:
            # label = ttp.reshape(len(ttp),1)
            label = np.hstack((ttp.reshape(len(ttp), 1), np.asarray(sigma_tmp).reshape(len(sigma_tmp), 1)))
            np.asarray(sigma_tmp)
            train = np.hstack((np.asarray(final_loc), np.asarray(encode_tmp)))
            for i in range(2):
                train[:, i] = (train[:, i] - min(train[:, i])) / (max(train[:, i]) - min(train[:, i]))

        # self.distance = np.linalg.norm(train - self.BSLoc, axis=1).reshape(-1, 1)
        # train = np.hstack((train, self.distance))
        #
        # for i in range(len(train[0])):
        #     train[:, i] = (train[:, i] - min(train[:, i])) / (max(train[:, i]) - min(train[:, i]))

        self.train = train
        self.label = label

        return train, label

    def tile_dis(self, tmp):  # returns the mean and variance of RSRP for a given tile.
        newdata = []
        encode_data = []
        for k in range(len(tmp)):
            for i in range(tmp[k, -1].astype(int)):
                newdata.append(tmp[k, -2])
                encode_data.append(tmp[k, 2:(self.num_colums-2)])
        if self.isavg == True:
            tile_mu = np.mean(np.asarray(newdata))
        else:
            tile_mu = np.asarray(newdata)
            tile_encode = np.asarray(encode_data)
        tile_sigma = np.var(np.asarray(newdata), ddof=1)
        if np.isnan(tile_sigma):
            tile_sigma = 100
        tile_all = np.asarray(newdata)
        return tile_mu, tile_sigma, tile_encode[0, :]

    def Data_preprocessing(self):
        self.encode_data_load()
        self.data_sampling(self.encode_alldata[:, 0:self.num_colums-2], self.encode_alldata[:, self.num_colums-2:self.num_colums])
        train_data, test_data = self.data_splitting(self.train, self.label)

        train_dataset = GetLoader(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        test_dataset = GetLoader(test_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader, test_loader, train_dataset, test_dataset


class GetLoader(Dataset):
    def __init__(self, df):
        self.x_data = df[:, :-2]
        self.y_data = df[:, -2].reshape(len(df), 1)
        self.gt_sigma = df[:, -1].reshape(len(df), 1)
        self.length = len(self.y_data)

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        return self.x_data[index], self.y_data[index], self.gt_sigma[index]

    def __len__(self):
        return self.length


def weights_init(m):
    """
    Initialize weights normal distributed with sd = 0.01
    :param m: weight matrix
    :return: normal distributed weights
    """
    m.weight.data.normal_(0.0, 0.01)


# Create Fully Connected Network
class Gaussian_Network(nn.Module):

    min_p = 1e-11

    def __init__(self, input_dim, output_dim):
        """
        Initialization
        :param input_dim: dimensionality of input
        :param output_dim: dimensionality of output
        """
        super(Gaussian_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc_add1 = nn.Linear(100, 100)
        # self.fc_add2 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)

        self.fcMu = nn.Linear(50, output_dim)
        weights_init(self.fcMu)
        self.fcSigma = nn.Linear(50, output_dim)
        weights_init(self.fcSigma)

    def forward(self, x):
        """
        Forward pass of input
        :param x: input
        :return: mu, Sigma of resulting output distribution
        """
        mid1 = F.relu(self.fc1(x)) # 4 100
        mid_add1 = F.relu(self.fc_add1(mid1)) # 100 100
        # mid_add2 = F.relu(self.fc_add2(mid_add1))
        mid = F.relu(self.fc2(mid_add1))   # 100 50
        mu = self.fcMu(mid) # 50 1
        # Sigma determined with ELUs + 1 + p to ensure values > 0
        # small p > 0 avoids that Sigma == 0
        sigma = F.elu(self.fcSigma(mid)) + 1 + self.min_p # 50 1
        return mu, sigma



class Linear_Model(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the Linear Model
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.learning_rate = 0.001
        self.num_epochs = 800
        self.loss_all = []
        self.TestLoss_all = []
        self.MAEerror_all = []
        self.MAEerror_P_all = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.GaussianNLLLoss(reduction='mean')
        self.create_model()
        self.isTerminal = False
        self.isAdaptiveLR = False
        self.val_in_every_train_epoc = True
        self.isCalculatingKLinTest = False
        self.isEarlyStopping = True
        self.early_stopping = EarlyStopping(self.isTerminal)


    def create_model(self):
        self.model = Gaussian_Network(input_dim=self.input_dim, output_dim=self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.03 ** (epoch // 100))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, loader, isSaving, test_loader, isLoading, model_path="linear.pth"):
        """
        Train the model and save the parameters
        :param loader:
        :param model_save_path: saved file name
        :param isSaving: if saving the model?
        :return: Loss of every epoch
        """

        self.model.train()
        for epoch in range(self.num_epochs):
            if self.isAdaptiveLR:
                self.adjust_learning_rate(epoch)
            loss_per_epoch = 0.0
            num_batches = 0
            # for x_data, y_data in loader:
            for x_data, y_data, gt_sigma in loader:
                # Get data to cuda if possible
                x_data = x_data[:,0:self.input_dim].to(device=self.device)
                y_data = y_data.to(device=self.device)

                # Get to correct shape
                x_data = x_data.reshape(x_data.shape[0], -1).float()
                # forward
                model_mu, model_sigma = self.model(x_data)
                loss = self.criterion(model_mu, y_data, model_sigma)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()
                loss_per_epoch += loss.item()
                num_batches += 1
                # MLpar_mean = np.mean(model_mu.detach().cpu().numpy())
                # MLpar_var  = np.mean(model_sigma.detach().cpu().numpy())
            averageloss = loss_per_epoch/num_batches

            # if epoch % 5 == 0:
            if self.isTerminal:
                print(f'training in epoch {epoch}, training loss is {averageloss}')
            # wandb.log({"train_loss_per_epoch": averageloss})
            if self.val_in_every_train_epoc:
                testLoss, MAEerror, MAEerror_P, m_r, m_p, v_r, v_p = self.test(test_loader, isLoading, model_path=model_path)
                # wandb.log({"test_loss_per_epoch": testLoss})
                self.loss_all.append(averageloss)
                self.TestLoss_all.append(testLoss)
                self.MAEerror_all.append(MAEerror)
                self.MAEerror_P_all.append(MAEerror_P)

                self.early_stopping(testLoss)
                if self.early_stopping.early_stop:
                    if isSaving:
                        torch.save(self.model.state_dict(), model_path)
                    break
        
        return self.loss_all, self.TestLoss_all, self.MAEerror_all, self.MAEerror_P_all


    def test(self, loader, isLoading, model_path="linear.pth"):
        """
        load the saved the parameters and test
        :param loader:
        :param isLoading: is loading trained model?
        :param model_path:  model name
        :return: None
        """
        if isLoading:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        loss_per_epoch = 0.0
        mu_all = []
        sigma_all = []
        loss_all = []
        mean_error_all = []
        mean_error_percentage_all = []
        KL_all = []
        num_batches = 0
        with torch.no_grad():
            # for x_data, y_data in loader:
            for x_data, y_data, gt_sigma in loader:
                x_data = x_data[:, 0:self.input_dim].to(device=self.device)
                y_data = y_data.to(device=self.device)

                x_data = x_data.reshape(x_data.shape[0], -1).float()

                model_mu, model_sigma = self.model(x_data)
                loss = self.criterion(model_mu, y_data, model_sigma)
                loss_per_epoch += loss.item()
                num_batches += 1
                mean_error_all.append(abs(model_mu.cpu().numpy() - y_data.cpu().numpy()).mean())
                mean_error_percentage_all.append((abs((model_mu.cpu().numpy() - y_data.cpu().numpy()) / y_data.cpu().numpy())).mean())
                if self.isCalculatingKLinTest:
                    mu_all.append(model_mu.cpu().numpy())
                    sigma_all.append(model_sigma.cpu().numpy())
                    loss_all.append(loss_per_epoch)
                    np_y_data = y_data.cpu().numpy()
                    np_model_mu = model_mu.cpu().numpy()
                    np_gt_sigma = gt_sigma.cpu().numpy()
                    np_model_sigma = model_sigma.cpu().numpy()
                    for i in range(len(np_y_data)):
                        KL = self.gaussian_KL_divergence(np_y_data[i], np_model_mu[i], np_gt_sigma[i], np_model_sigma[i])
                        KL_all.append(KL)
            averageloss = loss_per_epoch / num_batches
            mean_error = np.asarray(mean_error_all).mean()
            mean_error_P = np.asarray(mean_error_percentage_all).mean()

        if self.isCalculatingKLinTest:
            np_KL = np.asarray(KL_all)
            x1 = np.delete(np_KL, np.where(np.isinf(np_KL))[0]).mean()
            print(f'mean error of test set is {mean_error}, KL divergence of test set is {x1}')
        else:
            if self.isTerminal:
                print(f'MAE test set is {mean_error}, MAE in percentage is {mean_error_P}, loss of test set is {averageloss}')
        # return loss_all, mu_all, sigma_all, mean_error
        return averageloss, mean_error, mean_error_P, y_data.cpu().numpy(), model_mu.cpu().numpy(), gt_sigma.cpu().numpy(), model_sigma.cpu().numpy()

    def plt_results(self, isShow = False):
        if isShow:
            plt.figure()
            plt.plot(self.loss_all)
            plt.show()

    def gaussian_KL_divergence(self, m1, m2, v1, v2):
        """

        :param m1: mean value of real gaussian distribution
        :param m2: mean value of predicted gaussian distribution
        :param v1: variance value of real gaussian distribution
        :param v2: variance value of predicted gaussian distribution
        :return: KL divergence
        """
        return np.log(v2/v1)+np.divide(np.add(np.square(v1), np.square(m1-m2)), 2*np.square(v2)) - 0.5

class Linear_Model_Var(Linear_Model):
    # remove the feature from DT
    def __init__(self, input_dim):
        super().__init__(input_dim)

    def train(self, loader, isSaving, test_loader, isLoading, model_path="linear.pth"):
        """
        Train the model and save the parameters
        :param loader:
        :param model_save_path: saved file name
        :param isSaving: if saving the model?
        :return: Loss of every epoch
        """

        self.model.train()
        for epoch in range(self.num_epochs):
            if self.isAdaptiveLR:
                self.adjust_learning_rate(epoch)
            loss_per_epoch = 0.0
            num_batches = 0
            # for x_data, y_data in loader:
            for x_data, y_data, gt_sigma in loader:
                # Get data to cuda if possible
                # x_data = x_data.to(device=self.device)
                x_data = x_data[:, 0:self.input_dim].to(device=self.device)
                y_data = y_data.to(device=self.device)

                # Get to correct shape
                x_data = x_data.reshape(x_data.shape[0], -1).float()
                # forward
                model_mu, model_sigma = self.model(x_data)
                loss = self.criterion(model_mu, y_data, model_sigma)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()
                loss_per_epoch += loss.item()
                num_batches += 1
                # MLpar_mean = np.mean(model_mu.detach().cpu().numpy())
                # MLpar_var  = np.mean(model_sigma.detach().cpu().numpy())
            averageloss = loss_per_epoch/num_batches

            # if epoch % 5 == 0:
            if self.isTerminal:
                print(f'training in epoch {epoch}, training loss is {averageloss}')
            # wandb.log({"train_loss_per_epoch": averageloss})

            if self.val_in_every_train_epoc:
                testLoss, MAEerror, MAEerror_P = self.test(test_loader, isLoading, model_path=model_path)
                # wandb.log({"test_loss_per_epoch": testLoss})
                self.loss_all.append(averageloss)
                self.TestLoss_all.append(testLoss)
                self.MAEerror_all.append(MAEerror)
                self.MAEerror_P_all.append(MAEerror_P)

                self.early_stopping(testLoss)
                if self.early_stopping.early_stop:
                    if isSaving:
                        torch.save(self.model.state_dict(), model_path)
                    break

        return self.loss_all, self.TestLoss_all, self.MAEerror_all, self.MAEerror_P_all

    def test(self, loader, isLoading, model_path="linear.pth"):
        """
        load the saved the parameters and test
        :param loader:
        :param isLoading: is loading trained model?
        :param model_path:  model name
        :return: None
        """
        if isLoading:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        loss_per_epoch = 0.0
        mu_all = []
        sigma_all = []
        loss_all = []
        mean_error_all = []
        KL_all = []
        mean_error_percentage_all = []
        num_batches = 0
        with torch.no_grad():
            # for x_data, y_data in loader:
            for x_data, y_data, gt_sigma in loader:
                # x_data = x_data.to(device=self.device)
                x_data = x_data[:, 0:self.input_dim].to(device=self.device)
                y_data = y_data.to(device=self.device)


                x_data = x_data.reshape(x_data.shape[0], -1).float()

                model_mu, model_sigma = self.model(x_data)
                loss = self.criterion(model_mu, y_data, model_sigma)
                loss_per_epoch += loss.item()
                num_batches += 1
                mean_error_all.append(abs(model_mu.cpu().numpy() - y_data.cpu().numpy()).mean())
                mean_error_percentage_all.append((abs((model_mu.cpu().numpy() - y_data.cpu().numpy()) / y_data.cpu().numpy())).mean())
                if self.isCalculatingKLinTest:
                    mu_all.append(model_mu.cpu().numpy())
                    sigma_all.append(model_sigma.cpu().numpy())
                    loss_all.append(loss_per_epoch)
                    np_y_data = y_data.cpu().numpy()
                    np_model_mu = model_mu.cpu().numpy()
                    np_gt_sigma = gt_sigma.cpu().numpy()
                    np_model_sigma = model_sigma.cpu().numpy()
                    for i in range(len(np_y_data)):
                        KL = self.gaussian_KL_divergence(np_y_data[i], np_model_mu[i], np_gt_sigma[i], np_model_sigma[i])
                        KL_all.append(KL)
            averageloss = loss_per_epoch / num_batches
            mean_error = np.asarray(mean_error_all).mean()
            mean_error_P = np.asarray(mean_error_percentage_all).mean()
        if self.isCalculatingKLinTest:
            np_KL = np.asarray(KL_all)
            x1 = np.delete(np_KL, np.where(np.isinf(np_KL))[0]).mean()
            print(f'mean error of test set is {mean_error}, KL divergence of test set is {x1}')
        else:
            if self.isTerminal:
                print(f'MAE test set is {mean_error}, MAE in percentage is {mean_error_P}, loss of test set is {averageloss}')
        # return loss_all, mu_all, sigma_all, mean_error
        return averageloss, mean_error, mean_error_P


class model_frozen(Linear_Model):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        # self.num_epochs = 500
        # self.create_model()

    def create_model(self):
        self.model = Gaussian_Network(input_dim=self.input_dim, output_dim=self.output_dim).to(self.device)
        self.model.load_state_dict(torch.load("modelv2.pth"))
        for name, parameter in self.model.named_parameters():
            if 'fc1' in name or 'fc_add1' in name:
                parameter.requires_grad = False

        self.optimizer = optim.Adamax(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)




class encoded_data_processing_DataNoLoader(encoded_data_processing):
    def __init__(self, folder_add, usingDTdata, xynodes_all, encode_out):
        super(encoded_data_processing_DataNoLoader, self).__init__(folder_add, usingDTdata, xynodes_all, encode_out)

    def Data_preprocessing(self):
        self.encode_data_load()
        self.data_sampling(self.encode_alldata[:, 0:self.num_colums - 2],
                           self.encode_alldata[:, self.num_colums - 2:self.num_colums])
        train_data, test_data = self.data_splitting(self.train, self.label)

        return train_data, test_data




class Merge_data_processing(data_load_processing):
    def __init__(self, folder_list):
        super(Merge_data_processing, self).__init__(folder_list)

    def data_load(self, folder_index):
        df=pd.read_csv(folder_index)
        x_min = df.xmin.to_numpy()
        y_min = df.ymin.to_numpy()
        rsrp = df.rsrp.to_numpy()
        nodes = df[df.columns[-1]].to_numpy()
        # xy_all = np.vstack((x_min, y_min, rsrp, nodes)).T
        x = np.vstack((x_min, y_min)).T
        y= np.vstack((rsrp, nodes)).T
        self.x = x
        self.y = y

    def data_cleaning(self, x, y):
        # remove outlier by using Hampel function
        x = x.astype(float)
        good_index_x = self.remove_outlier_Hampel(x[:,0])
        mid_x = x[good_index_x, :]
        mid_y = y[good_index_x, :]

        good_index_y = self.remove_outlier_Hampel(mid_x[:,1])
        mid_x_final = mid_x[good_index_y, :]
        mid_y_final = mid_y[good_index_y, :]

        x = mid_x_final
        y = mid_y_final
        return x, y

    def Data_preprocessing(self):
        clean_x_all = []
        clean_y_all = []
        t = 0
        for x in self.folder:
            self.data_load(x)
            clean_x, clean_y = self.data_cleaning(self.x, self.y)

            self.nor_x = clean_x
            self.nor_y = clean_y

            train, label = self.data_sampling(self.nor_x, self.nor_y)
            train = np.column_stack((train, np.full((len(train), 1), t)))
            t = t + 1
            clean_x_all.append(train)
            clean_y_all.append(label)

        train = np.concatenate((clean_x_all[0],clean_x_all[1],clean_x_all[2]))
        label = np.concatenate((clean_y_all[0],clean_y_all[1],clean_y_all[2]))

        train[:, 0] = (train[:, 0] - min(train[:, 0])) / (max(train[:, 0]) - min(train[:, 0]))
        train[:, 1] = (train[:, 1] - min(train[:, 1])) / (max(train[:, 1]) - min(train[:, 1]))
        train[:, 2] = (train[:, 2] - min(train[:, 2])) / (max(train[:, 2]) - min(train[:, 2]))

        train_data, test_data = self.data_splitting(train, label)

        train_dataset = GetLoader(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        test_dataset = GetLoader(test_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader, test_loader
        # return tile_mu

class Merge_monthly_data_processing(Merge_data_processing):
    def __init__(self, folder_list):
        super(Merge_data_processing, self).__init__(folder_list)

    def Data_preprocessing(self):
        train_month = []
        label_month = []
        t = 0
        k = 0
        for x in self.folder:
            train, label = self.processOneMonth(x)
            train = np.column_stack((train, np.full((len(train), 1), k)))
            k = k+1
            train_month.append(train)
            label_month.append(label)
        train_all = np.concatenate((train_month[0], train_month[1]))
        label_all = np.concatenate((label_month[0], label_month[1]))

        for i in range(len(train_all[0])):
            train_all[:, i] = (train_all[:, i] - min(train_all[:, i])) / (max(train_all[:, i]) - min(train_all[:, i]))

        train_data, test_data = self.data_splitting(train_all, label_all)
        #
        train_dataset = GetLoader(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        test_dataset = GetLoader(test_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader, test_loader

    def processOneMonth(self, x):
        clean_x_all = []
        clean_y_all = []
        t = 0
        for ii in x:
            self.data_load(ii)
            clean_x, clean_y = self.data_cleaning(self.x, self.y)

            self.nor_x = clean_x
            self.nor_y = clean_y

            train1, label1 = self.data_sampling(self.nor_x, self.nor_y)
            train1 = np.column_stack((train1, np.full((len(train1), 1), t)))
            t = t + 1
            clean_x_all.append(train1)
            clean_y_all.append(label1)
        train = np.concatenate((clean_x_all[0], clean_x_all[1], clean_x_all[2]))
        # train = np.column_stack((train, np.full((len(train), 1), k)))
        label = np.concatenate((clean_y_all[0], clean_y_all[1], clean_y_all[2]))
        return train, label


