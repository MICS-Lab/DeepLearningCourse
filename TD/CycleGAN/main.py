import config
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import GlassesNoGlassesDataset
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
        gen_glasses,
        gen_no_glasses,
        disc_glasses,
        disc_no_glasses,
        loader,
        l1,
        mse,
        opt_gen,
        opt_disc,
):



def main():
    dataset = GlassesNoGlassesDataset(
        config.TRAIN_DIR + "glasses",
        config.TRAIN_DIR + "no_glasses",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(0.5, 0.5)
        ])
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    gen_glasses = Generator().to(config.DEVICE)
    gen_no_glasses = Generator().to(config.DEVICE)
    disc_glasses = Discriminator().to(config.DEVICE)
    disc_no_glasses = Discriminator().to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_glasses.parameters()) + list(gen_no_glasses.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999)
    )
    opt_disc = optim.Adam(
        list(disc_glasses.parameters()) + list(disc_no_glasses.parameters()),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    for epoch in config.NUM_EPOCHS:
        train_fn(
            gen_glasses,
            gen_no_glasses,
            disc_glasses,
            disc_no_glasses,
            loader,
            l1,
            mse,
            opt_gen,
            opt_disc,
        )


