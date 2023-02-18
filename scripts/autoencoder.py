#imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
import pickle
from sklearn.decomposition import PCA

torch.manual_seed(0)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# download and load MNIST dataset with torch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2**10,
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2**10,
                                        shuffle=False, num_workers=2)


# create an autoencoder to compress the MNIST dataset
for code in [2, 4, 8, 16, 32, 64, 128, 256][::-1]:
    torch.manual_seed(0)
    hidden_encoder = max(128, code*2)
    hidden_decoder = hidden_encoder
    print(f'code: {code}')
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(28*28, hidden_encoder),
                nn.ReLU(True),
                nn.Linear(hidden_encoder, code),
                nn.ReLU(True))
            self.decoder = nn.Sequential(
                nn.Linear(code, hidden_decoder),
                nn.ReLU(True),
                nn.Linear(hidden_decoder, 28*28),
                nn.Tanh())

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    try:
        net = torch.load(f"autoencoder-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pth")
        net.to(device)
        losses = pickle.load(f"autoencoder_losses-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pickle")
        print("loaded model")
    except:
        # initialize the autoencoder
        net = Autoencoder().to(device)
        # define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        # train the autoencoder
        n_epochs = 100
        losses = []
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, _ = data
                inputs = inputs.view(inputs.size(0), -1).to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss /= len(trainloader)
            losses.append(running_loss)
            print(f'[{epoch+1:2d}/{n_epochs}] loss: {running_loss:.5e}')
        torch.save(net, f"autoencoder-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pth")
        f = open(f"autoencoder_losses-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pickle", "wb")
        pickle.dump(losses, f)
        f.close()
    
    # test the autoencoder
    criterion = nn.MSELoss()
    total_loss = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).to(device)# flatten the inputs
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()
    total_loss /= len(testloader)
    print(f'loss: {total_loss:.3f}')

    # visualize the results
    fig, axes = plt.subplots(figsize=(15,7) , nrows=4, ncols=1)
    fig.suptitle(f"{28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28}", fontsize=16)
    axes[0].set_title("Originals", fontsize=14)
    axes[1].set_title("Reconstructed", fontsize=14)
    axes[2].set_title("Training", fontsize=14, y=0.9)
    for ax in axes:
        ax.axis('off')
    # show the original images
    for i in range(10):
        ax = fig.add_subplot(4, 10, i+1)
        ax.imshow(inputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the reconstructed images
    for i in range(10):
        ax = fig.add_subplot(4, 10, i+11)
        ax.imshow(outputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # plot the loss
    ax = fig.add_subplot(2, 3, 5)
    ax.semilogy(losses)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.savefig(f"AE-{code}.svg")
    plt.show()
    


    ########################################
    # create a PCA model
    pca = PCA(n_components=code)
    images_train = trainset.data.reshape(-1, 28*28).numpy()
    z_train = pca.fit_transform(images_train)
    # test PCA
    images_test = testset.data[-10:].reshape(-1, 28*28)
    images_pca = pca.transform(images_test)
    images_recon = pca.inverse_transform(images_pca)

    # visualize the results
    fig, axes = plt.subplots(figsize=(15,5) , nrows=2, ncols=1)
    axes[0].set_title("Originals", fontsize=14)
    axes[1].set_title(f"PCA Reconstructed ({code} components)", fontsize=14)
    for ax in axes:
        ax.axis('off')
    # show the original images
    for i in range(10):
        ax = fig.add_subplot(2, 10, i+1)
        ax.imshow(images_test[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    # show the PCA images
    for i in range(10):
        ax = fig.add_subplot(2, 10, i+11)
        ax.imshow(images_recon[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"PCA-{code}.svg", bbox_inches='tight')
    plt.show()


    ########################################
    # compare with autoencoder
    # PCA
    images_test = testset.data[:10].reshape(-1, 28*28)
    images_pca = pca.transform(images_test)
    images_recon = pca.inverse_transform(images_pca)
    # AE
    total_loss = 0
    for data in testloader:
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).to(device)# flatten the inputs
        outputs = net(inputs)
        break

    fig, axes = plt.subplots(figsize=(15,6) , nrows=3, ncols=1)
    axes[0].set_title("Originals", fontsize=14)
    axes[1].set_title(f"Reconstructed ({28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28})", fontsize=14)
    axes[2].set_title(f"PCA ({code} components)", fontsize=14)
    for ax in axes:
        ax.axis('off')
    # show the original images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+1)
        ax.imshow(inputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the reconstructed images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+11)
        ax.imshow(outputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the PCA images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+21)
        ax.imshow(images_recon[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"AE-PCA-{code}.svg", bbox_inches='tight')
    plt.show()


    ########################################
    # generate new images with AE
    z = torch.randn(100, code).to(device)
    new_images = net.decoder(z)

    fig, axes = plt.subplots(figsize=(15,4) , nrows=1, ncols=1)
    axes.set_title(f"Generated Images ({code} → {hidden_decoder} → {28*28})", fontsize=14)
    axes.axis('off')
    # show the original images
    for i in range(30):
        ax = fig.add_subplot(3, 10, i+1)
        ax.imshow(new_images.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"AE-Generated-{code}.svg", bbox_inches='tight')
    plt.show()

    ########################################
    # generate new images with PCA
    mean = z_train.mean(axis=0)
    std = z_train.std(axis=0)
    z = (torch.randn(100, code)*std + mean).numpy()
    new_images = pca.inverse_transform(z)

    fig, axes = plt.subplots(figsize=(15,5) , nrows=1, ncols=1)
    axes.set_title(f"PCA Generated Images ({code} components)", fontsize=14)
    axes.axis('off')
    # show the original images
    for i in range(30):
        ax = fig.add_subplot(3, 10, i+1)
        ax.imshow(new_images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"PCA-Generated-{code}.svg", bbox_inches='tight')
    plt.show()

    ########################################
    
    if code==2:
        # plot the latent space
        fig, axes = plt.subplots(figsize=(10,10) , nrows=1, ncols=1)
        axes.set_title(f"Latent Space ({code} → {hidden_encoder} → {code})", fontsize=14)
        axes.axis('off')
        # show the original images
        for i in range(10):
            ax = fig.add_subplot(10, 10, i+1)
            ax.imshow(testset.data[i].reshape(28, 28), cmap='gray')
            ax.axis('off')

        # show the latent space
        for i in range(10):
            ax = fig.add_subplot(10, 10, i+11)
            ax.imshow(outputs.cpu().data[i].view(28, 28), cmap='gray')
            ax.axis('off')
        plt.savefig(f"AE-Latent-{code}.svg", bbox_inches='tight')
