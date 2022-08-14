import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from nn_utilities import Linear_Model, Linear_Model_Var, GetLoader, encoded_data_processing_DataNoLoader, reset_weights, encoded_data_processing
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import KFold
# from Serve_DP import plot_results
# wandb.init(project="DT_ML")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# import folders
# folder = '../data/bris_mdt_map_10231C11_10m.csv'
# folder_Jan_A = '../data/MDT data_jan/mdt_10231A11_10m_jan.csv'
# folder_Jan_B = '../data/MDT data_jan/mdt_10231B11_10m_jan.csv'
# folder_Jan_C = '../data/MDT data_jan/mdt_10231C11_10m_jan.csv'
# folder_Aug_A = '../data/MDT data_aug/mdt_10231A11_10m_aug.csv'
# folder_Aug_B = '../data/MDT data_aug/mdt_10231B11_10m_aug.csv'
# folder_Aug_C = '../data/MDT data_aug/mdt_10231C11_10m_aug.csv'

month = "aug"
sector = "C"

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

    if month == "jan":
        if sector == "A":
            xynodes_all = [scio.loadmat('../VAE/matlab_data/xy_without_outliers/jan/tmp1A.mat')['tmp1']]
        else:
            xynodes_all = [scio.loadmat(f'../VAE/matlab_data/xy_without_outliers/jan/tmp1{sector}.mat')[f'tmp1{sector}']]
        encode_all = [np.load(f"../VAE/encoded_output_z/jan/encode_output_sector_{sector}_1403_zeta.npy")]
    elif month == "aug":
        xynodes_all = [scio.loadmat(f'../VAE/matlab_data/xy_without_outliers/{month}/tmp1{sector}_{month}.mat')[f'tmp1{sector}']]
        encode_all = [np.load(f"../VAE/encoded_output_z/{month}/encode_output_sector_{sector}_0704_{month}_zeta.npy")]

    folder_month_sector = f'../data/MDT data_{month}/mdt_10231{sector}11_10m_{month}.csv'

    Data = encoded_data_processing(folder_month_sector, False, xynodes_all[0], encode_all[0])
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

        input_dim = 4 # x,y,DT(mean,variance)
        model_DT = Linear_Model(input_dim)
        model_DT.apply(reset_weights)
        trainloss, testloss, MAError, MAError_P = model_DT.train(trainloader, isSaving=False, test_loader=testloader, isLoading=False, model_path="./modelReal_encode.pth")
 
        input_dim = 2  # x,y
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


    np.save(f"train_test_output/npy/{month}/{sector}/dim3Loss_all_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim3Loss_all)
    np.save(f"train_test_output/npy/{month}/{sector}/dim2Loss_all_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim2Loss_all)
    np.save(f"train_test_output/npy/{month}/{sector}/data_needed_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", data_needed)
    np.save(f"train_test_output/npy/{month}/{sector}/MAError_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim3MAError_all)
    np.save(f"train_test_output/npy/{month}/{sector}/MAError_var_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim2MAError_all)
    np.save(f"train_test_output/npy/{month}/{sector}/MAError_P_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim3MAError_P_all)
    np.save(f"train_test_output/npy/{month}/{sector}/MAError_P_var_10flod_{sector}_v2_1403_{month}_vae2_4f.npy", dim2MAError_P_all)

    # note: dim3Loss_all_10flod_A_v2_1403_vae2.npy shows improvement around 5%
    # plot_results(dim3MAError_all.T, dim2MAError_all.T, dim3MAError_P_all.T, dim2MAError_P_all.T, dim3Loss_all.T, dim2Loss_all.T, data_needed)
