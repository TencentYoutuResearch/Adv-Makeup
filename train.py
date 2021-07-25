# -*- coding: utf-8 -*-
from config import *
from model import *
from dataset import *

import sys
import os
import torchvision

config = Configuration()
device = torch.device('cuda:{}'.format(config.gpu)) if config.gpu >= 0 else torch.device('cpu')

def model_save(model, ep, target_name):
    if (ep + 1) % 50 == 0:
        os.makedirs(os.path.join(config.model_dir, target_name.split('.')[0]), exist_ok=True)
        print('--- save the model @ ep %d ---' % (ep))
        # Save the params of encoder in the generator
        torch.save(model.enc.state_dict(),
                   '%s/%05d_enc.pth' % (os.path.join(config.model_dir, target_name.split('.')[0]), ep))
        # Save the params of decoder in the generator
        torch.save(model.dec.state_dict(),
                   '%s/%05d_dec.pth' % (os.path.join(config.model_dir, target_name.split('.')[0]), ep))
        # Save the params of discriminator
        torch.save(model.discr.state_dict(),
                   '%s/%05d_discr.pth' % (os.path.join(config.model_dir, target_name.split('.')[0]), ep))

def train_single_epoch(model, train_loader, target_name):
    for it, data in enumerate(train_loader):
        # Un-makeup eye area images (attacker's)
        before_img = data[0].to(device).detach()
        # Real-world makeup eye area images
        after_img = data[1].to(device).detach()
        # Un-makeup images' path
        before_path = data[2]

        # Update the eye makeup discriminator
        model.update_discr(before_img, after_img)
        # Update the eye makeup generator
        model.update_gen(before_img, before_path, target_name)

def train(model, train_loader, target_name):
    for ep in range(config.epoch_steps):
        # Re-init all the params for each training epoch
        model.res_init(ep)

        # Update a single epoch
        train_single_epoch(model, train_loader, target_name)

        # Save the visualized images during training and output the training logs
        model.visualization(ep, len(train_loader))

        # Save model
        model_save(model, ep, target_name)


def main():
    for target_name in os.listdir(config.data_dir + '/target_aligned_600'):
        print("Target: %s" % (target_name))
        # Initialize the Adv-Makeup, networks and optimizers
        model = MakeupAttack(config)
        # Initialize the data-loader
        dataset = dataset_makeup(config)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=config.n_threads)
        train(model, train_loader, target_name)

if __name__ == '__main__':
    main()
