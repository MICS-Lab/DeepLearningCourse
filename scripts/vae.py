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

torch.manual_seed(0)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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


# create an variational autoencoder to compress the MNIST dataset
for code in [4, 8, 16, 32, 64, 128, 256][::-1]:
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
        v_net = torch.load(f"variational_autoencoder-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pth")
        v_net.to(device)
        f = open(f"variational_autoencoder_losses-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pickle", "rb")
        losses = pickle.load(f)
        f.close()
        reconstruction_losses, distribution_losses = losses
        print("loaded model")
    except:
        # initialize the autoencoder
        v_net = Autoencoder().to(device)
        # define the loss function and optimizer
        reconstruction_criterion = nn.MSELoss()
        target_distribution = torch.distributions.Normal(0,1)
        beta = 1
        distribution_criterion = lambda z: -beta * target_distribution.log_prob(z).mean()
        optimizer = torch.optim.Adam(v_net.parameters(), lr=1e-3)
        # train the autoencoder
        n_epochs = 100
        reconstruction_losses = []
        distribution_losses = []
        for epoch in range(n_epochs):
            running_reconstruction_loss = 0.0
            running_distribution_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, _ = data
                inputs = inputs.view(inputs.size(0), -1).to(device)
                optimizer.zero_grad()
                outputs = v_net(inputs)
                reconstruction_loss = reconstruction_criterion(outputs, inputs)
                z = v_net.encoder(inputs)
                distribution_loss = distribution_criterion(z)
                loss = reconstruction_loss + distribution_loss
                loss.backward()
                optimizer.step()
                running_reconstruction_loss += reconstruction_loss.item()
                running_distribution_loss += distribution_loss.item()
            running_reconstruction_loss /= len(trainloader)
            running_distribution_loss /= len(trainloader)
            reconstruction_losses.append(running_reconstruction_loss)
            distribution_losses.append(running_distribution_loss)
            print(f'[{epoch+1:2d}/{n_epochs}] reconstruction loss: {running_reconstruction_loss:.5e} distribution loss: {running_distribution_loss:.5e}')
        torch.save(v_net, f"variational_autoencoder-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pth")
        losses = [reconstruction_losses, distribution_losses]
        f = open(f"variational_autoencoder_losses-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pickle", "wb")
        pickle.dump(losses, f)
        f.close()

    # test the variational autoencoder
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).to(device)# flatten the inputs
        outputs = v_net(inputs)

    # visualize the results
    fig, axes = plt.subplots(figsize=(15,7) , nrows=4, ncols=1)
    fig.suptitle(f"{28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28}", fontsize=16)
    axes[0].set_title("Originals", fontsize=14)
    axes[1].set_title("Reconstructed", fontsize=14)
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
    ax = fig.add_subplot(2, 2, 3)
    ax.semilogy(reconstruction_losses)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Reconstruction loss')
    ax = fig.add_subplot(2, 2, 4)
    ax.semilogy(distribution_losses)
    ax.set_xlabel('epoch')
    ax.set_title('Distribution loss')
    plt.savefig(f"VAE-{code}.svg")
    plt.show()


    ########################################
    # compare VAE and AE
    net = torch.load(f"autoencoder-{28*28}_{hidden_encoder}_{code}-{code}_{hidden_decoder}_{28*28}.pth")
    net.to(device)
    # test the autoencoder
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).to(device)
        outputs = net(inputs)
    # test the variational autoencoder
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).to(device)
        v_outputs = v_net(inputs)
    
    fig, axes = plt.subplots(figsize=(15,6) , nrows=3, ncols=1)
    axes[0].set_title("Originals", fontsize=14)
    axes[1].set_title(f"AE Reconstructed ({28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28})", fontsize=14)
    axes[2].set_title(f"VAE Reconstructed  ({28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28})", fontsize=14)
    for ax in axes:
        ax.axis('off')
    # show the original images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+1)
        ax.imshow(inputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the AE reconstructed images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+11)
        ax.imshow(outputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the VAE reconstructed images
    for i in range(10):
        ax = fig.add_subplot(3, 10, i+21)
        ax.imshow(v_outputs.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"AE-VAE-{code}.svg", bbox_inches='tight')
    plt.show()

    ########################################
    # generate new images with AE
    z = torch.randn(100, code).to(device)
    new_images = v_net.decoder(z)

    fig, axes = plt.subplots(figsize=(15,4) , nrows=1, ncols=1)
    axes.set_title(f"Generated Images ({code} → {hidden_decoder} → {28*28})", fontsize=14)
    axes.axis('off')
    # show the original images
    for i in range(30):
        ax = fig.add_subplot(3, 10, i+1)
        ax.imshow(new_images.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"VAE-Generated-{code}.svg", bbox_inches='tight')
    plt.show()

    ########################################
    # generate new images with VAE & AE
    z = torch.randn(100, code).to(device)
    new_images = net.decoder(z)
    v_new_images = v_net.decoder(z)

    fig, axes = plt.subplots(figsize=(15,4) , nrows=2, ncols=1)
    axes[0].set_title(f"AE Generated ({28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28})", fontsize=14)
    axes[1].set_title(f"VAE Generated  ({28*28} → {hidden_encoder} → {code} ⇝ {code} → {hidden_decoder} → {28*28})", fontsize=14)
    for ax in axes:
        ax.axis('off')
    # show the AE generated images
    for i in range(10):
        ax = fig.add_subplot(2, 10, i+1)
        ax.imshow(new_images.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    # show the VAE generated images
    for i in range(10):
        ax = fig.add_subplot(2, 10, i+11)
        ax.imshow(v_new_images.cpu().data[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.savefig(f"AE-VAE-Generated-{code}.svg", bbox_inches='tight')
    plt.show()

    ########################################
    plt.close()