import numpy as np
import torch
from torchvision import datasets, transforms
from VAE_simple import VAE, VAE_encoder
import os
import scipy.io as scio

from torch.utils.data import Dataset
from PIL import Image

def extractor(model_path, path_dir, xy_loc, tmp1, savedName, verbose):

    xx = np.ones((len(tmp1), 1), dtype=int)
    for i in range(len(tmp1)):
        xx[i] = np.where((tmp1[i, 0] == xy_loc[:, 0]) & (tmp1[i, 1] == xy_loc[:, 1])==True)

    encode_model = VAE_encoder()
    save_model = torch.load(model_path)
    model_dict = encode_model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    encode_model.load_state_dict(model_dict)
    encode_model.eval()
    encode_model.to(device)

    transform = transforms.Compose([transforms.Resize([256, 256]),
                                          # transforms.Grayscale(num_output_channels=1),
                                          # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])

    class MyDataset(Dataset):  # 继承Dataset
        def __init__(self, path_dir, transform=None):  # 初始化一些属性
            self.path_dir = path_dir  # 文件路径,如'.\data\cat-dog'
            self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
            self.images = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中
            # self.loc_index = loc_index

        def __len__(self):  # 返回整个数据集的大小
            return len(self.images)

        def __getitem__(self, index):  # 根据索引index返回图像及标签
            image_index = self.images[index]  # 根据索引获取图像文件名称
            img_path = os.path.join(self.path_dir, image_index)  # 获取图像的路径或目录
            img = Image.open(img_path).convert('RGB')  # 读取图像

            # 根据目录名称获取图像标签（cat或dog）
            # label = img_path.split('\\')[-1].split('.')[0]
            # 把字符转换为数字cat-0，dog-1
            # label = 1 if 'dog' in label else 0

            if self.transform is not None:
                img = self.transform(img)
            return img

    dataset = MyDataset(path_dir,transform=transform)


    encode_out = np.ones((len(xx), 2))

    for k in range(len(xx)):
        tmp_data = dataset[xx[k, 0]].unsqueeze(0).to(device)
        zeta = encode_model(tmp_data)
        encode_out[k, 0:2]= zeta.detach().cpu().numpy()
        # encode_out[k, 2:4]= logvar.detach().cpu().numpy()

        if verbose:
            if k % 400 == 0:
                print(f'k is {k}, encoded data is mean_e: {encode_out[k, 0]},logvar_e: {encode_out[k, 1]}') # latent variable mean_e. logvar_e
    np.save(savedName, encode_out)
    return encode_out

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "SavedModel/encode/DT_2D_deep_dp_adam_workable_40_color_147190.1094.pt"
    path_dir = '../DRIVE/Drive_output/unzipped/test1/encoder_B_v2'  # section B
    xy_loc = scio.loadmat('matlab_data/xy_location/xy_loc_B.mat')['xy_loc_B']  # xy geographical location of sector B
    # xy_loc = scio.loadmat('matlab_data/xy_location/xy_loc_A.mat')['xy_loc']  # xy geographical location of sector A
    # tmp1 = scio.loadmat('matlab_data/xy_without_outliers/jan/tmp1A.mat')['tmp1']  # all the data after removing outlier from sector A
    tmp1 = scio.loadmat('matlab_data/xy_without_outliers/jan/tmp1B.mat')['tmp1B']  # all the data after removing outlier from sector B
    savedName = f'encoded_output_z/jan/encode_output_sector_B_1403_zeta.npy'
    extractor(model_path, path_dir, xy_loc, tmp1, savedName, verbose=True)
