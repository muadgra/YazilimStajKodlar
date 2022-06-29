# -*- coding: utf-8 -*-
"""
@author: Mertcan
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as opt

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root = './mnistdata', train = True, download = True, transform = transform)
data_loader = torch.utils.data.DataLoader(dataset = mnist_data,
                                          batch_size = 64,
                                          shuffle = True)

#Analyse the values before setting activation functions.
#In this case, values are between 0 and 1.
'''
data_iter = iter(data_loader)
images, labels = data_iter.next()
print(torch.min(images), torch.max(images))
'''

class AutoEncoder(nn.Module):
    def __init__(self):
        #(batch size, 1, 28, 28)
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride = 2, padding = 1), # (batch size, 16, 14, 14)
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride = 2, padding = 1), #(batch size, 32, 7, 7)
                nn.ReLU(),
                nn.Conv2d(32, 64, 7), #(batch size, 64, 1, 1)
        )
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 7),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride = 2, padding = 1, output_padding = 1), #(batch size, 16, 14, 14)
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride = 2, padding = 1, output_padding = 1), #(batch size, 1, 28, 28)
                #image pixel values are between 0 and 1
                #in case of [-1, 1], use tanh
                nn.Sigmoid()
            )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = opt.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        reconstructed = model(img)
        loss = criterion(reconstructed, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    outputs.append((epoch, img, reconstructed))



for i in range(0, num_epochs, 4):
    plt.figure(figsize = (9, 2))
    plt.gray()
    imgs = outputs[i][1].detach().numpy()
    recon = outputs[i][2].detach().numpy()
    for k, item in enumerate(imgs):
        if k >= 9: break
        plt.subplot(2, 9, k + 1)
        plt.imshow(item[0])
        
    for k, item in enumerate(recon):
        if k >= 9: break
        plt.subplot(2, 9, 9 + k + 1)
        plt.imshow(item[0])
