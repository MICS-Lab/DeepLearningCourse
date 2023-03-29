import config
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import GlassesNoGlassesDataset
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
        epoch,
        gen_glasses,
        gen_no_glasses,
        disc_glasses,
        disc_no_glasses,
        loader,
        val_loader,
        l1,
        mse,
        opt_gen,
        opt_disc,
):
    for idx, (glasses, no_glasses) in enumerate(loader):
        ## PRINT ADVANCEMENT
        print(f"{idx} / {len(loader)}")

        ## DISPLAY WHAT THE NETWORK IS CURRENTLY ABLE TO DO
        if idx % 100 == 0:
            with torch.no_grad():
                my_images = []
                for val_glasses, val_no_glasses in val_loader:
                    val_glasses = val_glasses.to(config.DEVICE)
                    val_no_glasses = val_no_glasses.to(config.DEVICE)
                    val_fake_glasses = gen_glasses(val_no_glasses)
                    val_fake_no_glasses = gen_no_glasses(val_glasses)

                    my_images_temp = [val_glasses, val_fake_no_glasses, val_no_glasses, val_fake_glasses]
                    for image_idx in range(len(my_images_temp)):
                        my_images_temp[image_idx] = my_images_temp[image_idx] * 0.5 + 0.5
                    my_images += my_images_temp

                # Create folder images/epoch if it doesn't already exist
                if not os.path.exists(f"{config.FOLDER_FOR_VAL_PLOTS}/{epoch}"):
                    os.makedirs(f"{config.FOLDER_FOR_VAL_PLOTS}/{epoch}")

                save_image(
                    torch.cat(my_images, dim=0),
                    f"{config.FOLDER_FOR_VAL_PLOTS}/{epoch}/val_{idx}.png"
                )

        ## TRAIN THE NETWORK
        # TRAIN THE GENERATOR
        # LOAD IMAGES (2), GENERATE FAKE IMAGES (2), GET DISCRIMINATOR OUTPUTS (2) NEEDED TO TRAIN THE GENERATOR
        glasses = glasses.to(config.DEVICE)
        # ^ vraie image de lunettes
        no_glasses = no_glasses.to(config.DEVICE)
        # ^ vraie image de pas lunettes
        fake_glasses = gen_glasses(no_glasses)
        # ^ image générée de lunettes
        fake_no_glasses = gen_no_glasses(glasses)
        # ^ image générée pas lunettes

        disc_fake_glasses = disc_glasses(fake_glasses)
        # ^ ce que pense le discriminateur de l'image générée de lunettes
        disc_fake_no_glasses = disc_no_glasses(fake_no_glasses)
        # ^ ce que pense le discrimnateur de l'image générée de pas lunettes

        loss_gen = mse(
            disc_fake_glasses,
            torch.ones_like(disc_fake_glasses) - random.random() * config.ONE_SIDED_LABEL_SMOOTHING
        )  # je veux que le discriminateur pense que mes images sont réelles
        loss_gen += mse(
            disc_fake_no_glasses,
            torch.ones_like(disc_fake_no_glasses) - random.random() * config.ONE_SIDED_LABEL_SMOOTHING
        )  # je veux que le discriminateur pense que mes images sont réelles
        loss_gen += config.LAMBDA_CYCLE * l1(
            gen_glasses(fake_no_glasses),  # j'enlève des lunettes et je rajoute des lunettes
            glasses
        )
        loss_gen += config.LAMBDA_CYCLE * l1(
            gen_no_glasses(fake_glasses),  # j'ajoute des lunettes et je les renlève
            no_glasses
        )

        gen_glasses.zero_grad()
        gen_no_glasses.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        # TRAIN THE DISCIMINATOR
        # LOAD IMAGES (2)*, GENERATE FAKE IMAGES (2)**, GET DISCRIMINATOR OUTPUTS (4) NEEDED TO TRAIN THE DISCRIMINATOR
        # * no need to reload them
        # ** no need to re-pass them in the generator
        # We re-compute disc_fake_glasses and disc_fake_no_glasses because we don't want to keep the computation
        # graph from the generator
        disc_fake_glasses = disc_glasses(fake_glasses.detach())
        # ^ ce que pense le discriminateur de l'image générée de lunettes
        disc_fake_no_glasses = disc_no_glasses(fake_no_glasses.detach())
        # ^ ce que pense le discrimnateur de l'image générée de pas lunettes
        disc_true_glasses = disc_glasses(glasses)
        # ^ ce que pense le discriminateur de la vraie image de lunettes
        disc_true_no_glasses = disc_no_glasses(no_glasses)
        # ^ ce que pense le discriminateur de la vraie image de lunettes
        loss_disc = mse(
            disc_fake_glasses,
            torch.zeros_like(disc_fake_glasses)
        )  # je veux être capable de me rendre compte que c'est une fausse image
        loss_disc += mse(
            disc_true_glasses,
            torch.ones_like(disc_true_glasses)
        )  # je veux être capable de me rendre compte que c'est une vraie image
        loss_disc += mse(
            disc_fake_no_glasses,
            torch.zeros_like(disc_fake_no_glasses) + random.random() * config.ONE_SIDED_LABEL_SMOOTHING
        )  # je veux être capable de me rendre compte que c'est une fausse image
        loss_disc += mse(
            disc_true_no_glasses,
            torch.ones_like(disc_true_no_glasses) + random.random() * config.ONE_SIDED_LABEL_SMOOTHING
        )  # je veux être capable de me rendre compte que c'est une vraie image

        disc_glasses.zero_grad()
        disc_no_glasses.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # PRINT LOSSES
        print(f"Loss Generator: {loss_gen.item():.4f}, Loss Discriminator: {loss_disc.item():.4f}")



def main():
    dataset = GlassesNoGlassesDataset(
        config.TRAIN_DIR + "/glasses",
        config.TRAIN_DIR + "/no_glasses",
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
    val_dataset = GlassesNoGlassesDataset(
        config.VAL_DIR + "/glasses",
        config.VAL_DIR + "/no_glasses",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize(0.5, 0.5)
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
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

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            epoch,
            gen_glasses,
            gen_no_glasses,
            disc_glasses,
            disc_no_glasses,
            loader,
            val_loader,
            l1,
            mse,
            opt_gen,
            opt_disc,
        )


if __name__ == "__main__":
    main()
