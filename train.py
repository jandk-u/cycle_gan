import torch

import config
from data import HorseZebraDataset
import sys
from utils import save_weight, load_weight
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
    step = 0
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # train Discriminator H and Z
        fake_horse = gen_H(zebra)
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        DH_loss = D_H_fake_loss + D_H_real_loss

        fake_zebra = gen_Z(horse)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        DZ_loss = D_Z_fake_loss + D_Z_real_loss

        D_loss = (DH_loss + DZ_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train Generators H and Z
        ### adversarial loss for both generator
        D_H_fake = disc_H(fake_horse)
        D_Z_fake = disc_Z(fake_zebra)
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

        ### cycle loss
        cycle_zebra = gen_Z(fake_horse)
        cycle_horse = gen_H(fake_zebra)
        cycle_zebra_loss = l1(zebra, cycle_zebra)
        cycle_horse_loss = l1(horse, cycle_horse)

        ### identity loss
        identity_zebra = gen_Z(zebra)
        identity_horse = gen_H(horse)
        identity_zebra_loss = l1(zebra, cycle_zebra)
        identity_horse_loss = l1(horse, cycle_horse)

        ## add all together
        G_loss = (
                loss_G_H + loss_G_Z +
                cycle_horse_loss*config.LAMBDA_CYCLE + cycle_zebra_loss*config.LAMBDA_CYCLE +
                identity_horse_loss*config.LAMBDA_IDENTITY + identity_zebra_loss*config.LAMBDA_IDENTITY
        )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if step % 200 == 0:
            save_image(fake_horse*0.5 + 0.5, f"saved_images/horse_{step}.jpg")
            save_image(fake_zebra*0.5 + 0.5, f"saved_images/zebra_{step}.jpg")

        step += 1


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_weight(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE
        )
        load_weight(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE
        )
        load_weight(
            config.CHECKPOINT_DISC_H, disc_H, opt_disc, config.LEARNING_RATE
        )
        load_weight(
            config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, config.LEARNING_RATE
        )

    data = HorseZebraDataset(root_zebra="/home/j/Dataset/horse2zebra/trainB",
                             root_horse="/home/j/Dataset/horse2zebra/trainA", transform=config.transforms)
    loader = DataLoader(dataset=data, batch_size=config.BATCH_SIZE, shuffle=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCH):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        if config.SAVE_MODEL:
            save_weight(gen_H, opt_gen, mse, epoch, path=config.CHECKPOINT_GEN_H)
            save_weight(gen_Z, opt_gen, mse, epoch, path=config.CHECKPOINT_GEN_Z)
            save_weight(disc_H, opt_disc, mse, epoch, path=config.CHECKPOINT_DISC_H)
            save_weight(disc_Z, opt_disc, mse, epoch, path=config.CHECKPOINT_DISC_Z)


if __name__ == '__main__':
    main()
