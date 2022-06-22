import utils
from losses import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1

        self.discriminator = discriminator.Discriminator(args, gan_type)
        
        if gan_type != 'WGAN_GP':
            self.optimizer = utils.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utils.make_scheduler(args, self.optimizer)

    def forward(self, fake, real, train=True):
          
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()

            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)         

            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)

            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
                    
            if train:
                # Discriminator update
                self.loss += loss_d.item()
                loss_d.backward()
                self.optimizer.step()

                if self.gan_type == 'WGAN':
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        if self.gan_type == 'GAN':
            d_fake_for_g = self.discriminator(fake)
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )

        elif self.gan_type.find('WGAN') >= 0:
            d_fake_for_g = self.discriminator(fake)
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g
