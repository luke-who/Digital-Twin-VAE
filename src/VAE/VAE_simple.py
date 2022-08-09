"""
AE_Simple.py
A simple version of autoencoder
"""
# PyTorch Imports
import torch.nn as nn
import torch
import torch.nn as nn
# from torch.autograd import Variable
import wandb


class VAE(nn.Module): # complex version of AE, good to use, but only 4 bits available
    # Initializer for the AE model
    def __init__(self):
        super(VAE, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        # -----------------------------------------------------------
        # Encoder
        # -----------------------------------------------------------

        # Convolutions and max-pooling
        self.en_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv1_1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.en_conv1_2 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        # self.en_conv1_inter = torch.cat((self.en_conv1, self.en_conv1_1, self.en_conv1_2), 1)

        self.en_max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.en_max1_indices = None
        self.en_max1_input_size = None

        self.en_conv2a = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        # self.en_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_max2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.en_max2_indices = None
        self.en_max2_input_size = None

        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_max3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.en_max3_indices = None
        self.en_max3_input_size = None

        # self.en_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Flattening
        self.en_maxflat_input_size = None
        self.en_maxflat_indices = None
        self.en_maxflat_output_size = None

        # Fully-connected layers
        self.en_fc1 = nn.Linear(128, 64)
        self.en_mean = nn.Linear(64, 2)
        self.en_var = nn.Linear(64, 2)

        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------

        # Fully-connected layers
        self.de_fc2 = nn.Linear(2, 64)
        self.de_fc1 = nn.Linear(64, 128)
        # self.de_fc1 = nn.Linear(128, 128)

        # Unflattening

        # De-convolutions and (un)max-pooling
        # self.de_deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.de_unmax3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.de_deconv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_deconv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.de_unmax2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # self.de_deconv2b = nn.ConvTranspose2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.de_deconv2a = nn.ConvTranspose2d(64, 192, kernel_size=3, stride=1, padding=1)

        self.de_unmax1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.de_deconv1 = nn.ConvTranspose2d(192, 3, kernel_size=3, stride=1, padding=1)


    # Encoder layers
    def encode(self, X):
        # Convolutions and max-pooling
        x1 = self.en_conv1(X) # 15 64 256 256
        x2 = self.en_conv1_1(X) # 15 64 256 256
        x3 = self.en_conv1_2(X) # 15 64 256 256
        x = torch.cat((x1, x2, x3), 1) # 15 192 256 256
        self.en_max1_input_size = x.size()  # 15 192 256 256
        x, self.en_max1_indices = self.en_max1(x) # x: 15 192 128 128, en_max1_indices: 15 192 128 128

        # x = self.en_conv2b(self.en_conv2a(x))
        x = self.en_conv2a(x) # 15 64 128 128

        self.en_max2_input_size = x.size() # 15 64 128 128
        x, self.en_max2_indices = self.en_max2(x) # 15 64 64 64; 15 64 64 64

        x = self.en_conv3a(x) # 15 128 64 64
        # x = self.en_conv3b(self.en_conv3a(x))

        self.en_max3_input_size = x.size() # 15 128 64 64
        x, self.en_max3_indices = self.en_max3(x) # 15 128 33 33; 15 128 33 33

        # x = self.en_conv4(x)

        # Flattening
        self.en_maxflat_input_size = x.size() # 15 128 33 33
        x, self.en_maxflat_indices = nn.MaxPool2d(x.size()[2:], return_indices=True)(x) # 15 128 1 1; 15 128 1 1
        x = x.view(-1, 128)
        x = self.dropout(x)
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        return self.en_mean(x), self.en_var(x)

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon*torch.exp(logvar/2)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        # return eps.mul(std).add_(mu)

    # Decoder layers
    def decode(self, z): # 15 2
        # Fully-connected layers
        # x = self.de_fc3(X)
        x = self.relu(self.de_fc2(z)) # 15 20
        # x = self.dropout(x)
        x = self.relu(self.de_fc1(x)) # 15 128
        # x = self.dropout(x)
        # Unflattening
        x = x.view(-1, 128, 1, 1) # 15 128 1 1
        x = nn.MaxUnpool2d(self.en_maxflat_input_size[2:])(x, self.en_maxflat_indices, self.en_maxflat_input_size) # 15 128 33 33

        # De-convolutions and (un)max-pooling
        # x = self.de_deconv4(x)

        x = self.de_unmax3(x, self.en_max3_indices, self.en_max3_input_size) # 15 128 64 64
        # x = self.de_deconv3a(self.de_deconv3b(x))
        x = self.de_deconv3a(x) # 15 64 64 64

        x = self.de_unmax2(x, self.en_max2_indices, self.en_max2_input_size) # 15 64 128 128
        x = self.de_deconv2a(x) # 15 192 128 128
        # x = self.de_deconv2a(self.de_deconv2b(x))

        x = self.de_unmax1(x, self.en_max1_indices, self.en_max1_input_size) # 15,192,256,256
        x = self.relu(self.de_deconv1(x)) # 15 3 256 256
        return x

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_encoder(VAE):
    def __init__(self):
        super(VAE_encoder, self).__init__()

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparametrize(mu, logvar)
        # return mu, logvar
        return z



