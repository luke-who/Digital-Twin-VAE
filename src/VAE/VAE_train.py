#%% import datasets
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import os
from nn_utilities import reset_weights
import pickle

from tqdm import tqdm
import torch.optim as optim
from VAE_simple import VAE
import datetime

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
# percentage of training set to use as validation
valid_size = 0.2
bottleneck = 2
# model = autoencoder(bottleneck)
model = VAE()
# model.apply(reset_weights)
# encoder = NN_encoder()
# decoder = NN_decoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

isUsingEncode = False
TrainingMachine = 'serve'
LoadExistingModel = False
UsingWandb = False
isSavingModel = True
isUsingValidation = True

if UsingWandb:
    import wandb
    wandb.init(project="DT_AE")

if TrainingMachine == 'local':
    data_dir = 'D:/OneDrive - University of Bristol/Images'
    # data_dir = 'D:/Images'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size = 15
    model_path = 'DT_Gray_2D_28_simpler_v2_Nomodule_0.0142.pt'
    print('Using local machine')
else:
    data_dir = '../DRIVE/Drive_output/unzipped'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    batch_size = 75
    # encoder = nn.DataParallel(encoder)
    # decoder = nn.DataParallel(decoder)
    if LoadExistingModel:
        model_path = 'SavedModel/encode/DT_2D_deep_dp_adam_workable_18_color_149467.4375.pt'
        save_model = torch.load(model_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    model = nn.DataParallel(model)
    print('Using cloud server')


# model.to(device)
# encoder.to(device)
# decoder.to(device)

train_transforms = transforms.Compose([
                                    #    transforms.Grayscale(),
                                    #    transforms.RandomRotation(45),
                                    #    transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                                       transforms.Resize([256, 256]),
                                       transforms.RandomVerticalFlip(),
                                       # transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       # transforms.Normalize(0.5, 0.5)])
                                    #    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    #                         std=[0.5, 0.5, 0.5])
                                        ])

# train_data = datasets.ImageFolder(data_dir + '/Train_SecAandB', transform=train_transforms)
# train_data = datasets.ImageFolder(data_dir + '/Train_smallsize_testonly', transform=train_transforms)
# train_data = datasets.ImageFolder(data_dir + '/SectorABC_ALL', transform=train_transforms)
train_data = datasets.ImageFolder(data_dir + '/ABCNewVersion', transform=train_transforms)

# obtain training indices that will be used for validation
num_train = len(train_data)
print(f'num of training data is {num_train}')
indices = list(range(num_train))

np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
# train_idx, valid_idx = indices[0:1000], indices[1000:1200]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)

n_epochs = 30

reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

optimizer = optim.Adam(model.parameters())

valid_loss_min = np.Inf  # model change in validation loss
trainloss = []
validloss = []


# def encoder_test():
#     # encode_model = NN_encoder()
#     # encode_model.apply(reset_weights)
#     # model_dict = encode_model.state_dict()
#     # state_dict = {k: v for k, v in model.module.state_dict().items() if k in model_dict.keys()}
#     # model_dict.update(state_dict)
#     # encode_model.load_state_dict(model_dict)
#     # encode_model.eval()
#     # encode_model.to(device)
#
#     outputs = []
#     i = 0
#     with torch.no_grad():
#         for data, _ in valid_loader:
#             data = data.to(device)
#             output = model(data)
#             # output_encode = encode_model(data)
#             # print(f'output of bottleneck is {output_encode[0]}')
#             # print(f'output of bottleneck at validation: bit0 is {output_encode[0, 0].item() :.5f}, bit1 is {output_encode[0, 1].item() :.5f}')
#             i += 1
#             break
#         outputs.append((data.data, output.data))
#     return outputs

def encoder_test():
    # encode_model = NN_encoder()
    # encode_model.apply(reset_weights)
    # model_dict = encode_model.state_dict()
    # state_dict = {k: v for k, v in model.module.state_dict().items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # encode_model.load_state_dict(model_dict)
    # encode_model.eval()
    # encode_model.to(device)

    outputs = []
    i = 0
    # encoder.eval()
    # decoder.eval()
    model.eval()
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            # decoder.encode(data)
            # decoded_data = decoder(encoded_data)
            # output_encode = encode_model(data)
            # print(f'output of bottleneck is {output_encode[0]}')
            # print(f'output of bottleneck at validation: bit0 is {output_encode[0, 0].item() :.5f}, bit1 is {output_encode[0, 1].item() :.5f}')
            i += 1
            break
        yal_loss = loss_function(recon_batch, data, mu, logvar)
        if UsingWandb:
            wandb.log({"validation loss": yal_loss.item()})
        outputs.append((data.data, recon_batch.data))
    return outputs


def plot_ae_outputs(outputs, imagesavename):
    k = 0
    num_subf = 1
    plt.figure(figsize=(10, 20))
    # plt.gray()
    imgs = outputs[k][0].cpu().detach().numpy()
    recon = outputs[k][1].cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= num_subf: break
        plt.subplot(2, num_subf, i + 1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        t = np.transpose(item, (1,2,0))
        plt.imshow((t * 255).astype(np.uint8))
    # plt.show()

    for i, item in enumerate(recon):
        if i >= num_subf: break
        plt.subplot(2, num_subf, num_subf + i + 1)  # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        t = np.transpose(item, (1, 2, 0))
        plt.imshow((t * 255).astype(np.uint8))
    if TrainingMachine == 'local':
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(imagesavename)
        plt.close()


# decoder.eval()
# with torch.no_grad():
#     for data, _ in valid_loader:
#         data = data.to(device)
#         decoder.encode(data)

loss_list = []
print(f'************Start training*************')
for epoch in range(n_epochs):
    train_loss = 0.0
    k1 = 0
    model.train()
    # encoder.train()
    # decoder.train()
    # print(f'Epoch:{epoch}')
    # count = 0
    bestloss = 0.0
    for data, _ in tqdm(train_loader):
        data = data.to(device)

        # output = model(data)
        # decoded_data = model(data)
        # decoder.module.encode(data)
        # wandb.log({'encoded data 4': encoded_data[0, 4].item()})
        # decoded_data = decoder(encoded_data)
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_loss += loss.item() * data.size(0)
        # print(f'training on batch {k1}')
        # k1 += 1
        loss_list.append(loss.item())

    print(f'Epoch:{epoch}, Loss:{loss.item():.4f}')
    if UsingWandb:
        wandb.log({"trianing loss": loss.item()})

    if isUsingValidation:
        outputs = encoder_test()

        name_encode = f'./SavedModel/encode/DT_2D_deep_dp_adam_workable_{epoch}_color_{loss.item():.4f}.pt'
        name_decode = f'./SavedModel/decode/DT_2D_deep_dp_adam_workable_{epoch}_color_{loss.item():.4f}.pt'
        imagesavename = f'./TrainingImage/Epoch_2D_deep_dp_adam_workable_{epoch}_{loss.item():.4f}.svg'
        if TrainingMachine == 'local':
            torch.save(encoder.state_dict(), name_encode)
            torch.save(decoder.state_dict(), name_decode)
        else:
            # torch.save(encoder.module.state_dict(), name_encode)
            # if loss.item() < bestloss:
            #     bestloss = loss.item()
            if isSavingModel:
                torch.save(model.module.state_dict(), name_encode)
            # torch.save(decoder.module.state_dict(), name_decode)
        plot_ae_outputs(outputs, imagesavename)

# with open('training_output/loss_list_normalize.pkl', 'wb') as f:
#     pickle.dump(loss_list, f)       
# with open('training_output/loss_list_rotation.pkl', 'wb') as f:
#     pickle.dump(loss_list, f)       
with open('training_output/loss_list_vertical_flip.pkl', 'wb') as f:
    pickle.dump(loss_list, f)    
f.close()



