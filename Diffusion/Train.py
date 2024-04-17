
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    modelConfig["save_weight_dir"] = os.path.join(modelConfig["save_weight_dir"], modelConfig['dataset'])
    # dataset
    if modelConfig['dataset']=='CIFAR10':
        dataset = CIFAR10(
            root='../dataset/CIFAR10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        modelConfig['ori_ch'] = 3
    elif modelConfig['dataset']=='mnist':
        dataset = MNIST(
            root='../dataset/mnist',
            train=True,
            download=True,
            transform=transforms.Compose([transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
            ])),
        modelConfig['ori_ch'] = 1
    dataloader = DataLoader(
        dataset[0], batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(ori_ch=modelConfig['ori_ch'], T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    modelConfig["save_weight_dir"] = os.path.join(modelConfig["save_weight_dir"], modelConfig['dataset'])
    modelConfig["sampled_dir"] = os.path.join(modelConfig["sampled_dir"], modelConfig['dataset'])
    if modelConfig['dataset']=='CIFAR10':
        modelConfig['ori_ch'] = 3
    elif modelConfig['dataset']=='mnist':
        modelConfig['ori_ch'] = 1

    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(ori_ch=modelConfig['ori_ch'], T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], modelConfig['ori_ch'], 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs, img_list = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

        #produce gif
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        random_index = 50

        fig = plt.figure()
        ims = []
        for im in img_list[::10]:
            im = im[random_index]
            im = torch.clip(im, -1, 1)
            im = im*0.5+0.5
            img = im.permute(1, 2, 0).detach().cpu().numpy()
            im = plt.imshow(img, animated=True)
            ims.append([im])

        animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        animate.save(os.path.join(modelConfig["sampled_dir"], 'diffusion.gif'), writer='pillow', fps=10)
        plt.show()