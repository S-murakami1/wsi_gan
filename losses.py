import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def discriminator_loss(D, real, fake):
    pred_real = D(real)
    pred_fake = D(fake.detach())

    loss_real = mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = mse_loss(pred_fake, torch.zeros_like(pred_fake))

    return 0.5 * (loss_real + loss_fake)


def generator_gan_loss(D, fake):
    pred_fake = D(fake)
    return mse_loss(pred_fake, torch.ones_like(pred_fake))


def cycle_consistency_loss(real, reconstructed):
    return l1_loss(reconstructed, real)
