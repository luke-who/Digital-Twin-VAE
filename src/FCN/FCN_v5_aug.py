import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from nn_utilities import Linear_Model, Linear_Model_Var, GetLoader, encoded_data_processing_DataNoLoader, reset_weights, encoded_data_processing
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import KFold
from Serve_DP import plot_results
# wandb.init(project="DT_ML")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import folders
folder = './Data/bris_mdt_map_10231C11_10m.csv'
folder_Jan_A = './Data/MDT data_jan/mdt_10231A11_10m_Jan.csv'
folder_Jan_B = './Data/MDT data_jan/mdt_10231B11_10m_Jan.csv'
folder_Jan_C = './Data/MDT data_jan/mdt_10231C11_10m_Jan.csv'
folder_Aug_A = './Data/MDT data_aug/mdt_10231A11_10m_aug.csv'
folder_Aug_B = './Data/MDT data_aug/mdt_10231B11_10m_aug.csv'
folder_Aug_C = './Data/MDT data_aug/mdt_10231C11_10m_aug.csv'

if __name__ == "__main__":
    k_folds = 20
    MAE_results = {}
    MAE_P_results = {}
    batch_size = 3000

    # Set fixed random number seed
    torch.manual_seed(42)

    dim3Loss_all = np.zeros((1, k_folds))
    dim2Loss_all = np.zeros((1, k_folds))
    dim2MAError_all = np.zeros((1, k_folds))
    dim3MAError_all = np.zeros((1, k_folds))
    data_needed = []
    dim2MAError_P_all = np.zeros((1, k_folds))
    dim3MAError_P_all = np.zeros((1, k_folds))


    xynodes_all = [scio.loadmat('tmp1C_aug.mat')['tmp1C']]
    encode_all = [np.load("encode_output_sector_C_0704_aug_zeta.npy")]

    Data = encoded_data_processing(folder_Aug_C, False, xynodes_all[0], encode_all[0])
    _, _, train_dataset, test_dataset = Data.Data_preprocessing()

    kfold = KFold(n_splits=k_folds, shuffle=True)
    dataset = ConcatDataset([train_dataset, test_dataset])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        input_dim = 4 # x,y,index,DT
        model_DT = Linear_Model(input_dim)
        model_DT.apply(reset_weights)
        trainloss, testloss, MAError, MAError_P = model_DT.train(trainloader, isSaving=False, test_loader=testloader, isLoading=False, model_path="./modelReal_encode.pth")

        input_dim = 2  # x,y,index,DT
        model_real = Linear_Model(input_dim)
        model_real.apply(reset_weights)
        trainloss_var, testloss_var, MAError_var, MAError_P_var = model_real.train(trainloader, isSaving=False, test_loader=testloader, isLoading=False, model_path="./modelReal_encode_var.pth")

        dim3Loss_all[0, fold] = (np.asarray(testloss)).min()
        dim2Loss_all[0, fold] = (np.asarray(testloss_var)).min()

        dim3MAError_all[0, fold] = (np.asarray(MAError)).min()
        dim2MAError_all[0, fold] = (np.asarray(MAError_var)).min()

        dim3MAError_P_all[0, fold] = (np.asarray(MAError_P)).min()
        dim2MAError_P_all[0, fold] = (np.asarray(MAError_P_var)).min()

        data_needed.append((trainloss, testloss, MAError, MAError_P, trainloss_var, testloss_var, MAError_var, MAError_P_var))


    np.save("./dim3Loss_all_10flod_C_v2_0704_aug_vae2_4f.npy", dim3Loss_all)
    np.save("./dim2Loss_all_10flod_C_v2_0704_aug_vae2_4f.npy", dim2Loss_all)
    np.save("./data_needed_10flod_C_v2_0704_aug_vae2_4f.npy", data_needed)
    np.save("./MAError_10flod_C_v2_1403_0704_aug_4f.npy", dim3MAError_all)
    np.save("./MAError_var_10flod_C_v2_0704_aug_vae2_4f.npy", dim2MAError_all)
    np.save("./MAError_P_10flod_C_v2_0704_aug_vae2_4f.npy", dim3MAError_P_all)
    np.save("./MAError_P_var_10flod_C_v2_0704_aug_vae2_4f.npy", dim2MAError_P_all)
