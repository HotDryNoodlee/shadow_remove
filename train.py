import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from data import create_dataset
from models import create_model
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from numpy import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io



@hydra.main(config_path="options", config_name="config")
def train(opt):
    print(opt)
    output = []
    max_psnr = 0
    max_ssim = 0
    train_dataset = create_dataset(opt.model.trainset)  
    test_dataset = create_dataset(opt.model.testset)
    model = create_model(opt.model)
    model.print_networks(model.opt.verbose)
    loss_dict = {name: [] for name in model.loss_names}


    for epoch in range(opt.model.epoch_count, opt.model.n_epoch + opt.model.n_epoch_decay + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)/opt.model.trainset.batch_size):
        # for i, data in enumerate(train_dataset):
            if epoch == opt.model.epoch_count and i == 0:
                if opt.model.model_name == "con":
                    model.data_dependent_initialize(data)
                model.setup()               # regular setup: load and print networks; create schedulers
                # import pdb;pdb.set_trace()
                # print(next(model.netG.parameters()).device)
                model.parallelize()
                # print(next(model.netG.parameters()).device)
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()
            if epoch > 1: 
                losses = model.get_current_losses()
                for key in losses:
                    loss_dict[key].append(losses[key])
        
        if epoch % opt.runtime.eval_frqe == 0:
            model.eval()
            # for i, data in enumerate(test_dataset):
            ssim = []
            psnr = []
            for i, data in tqdm(enumerate(test_dataset), total=len(test_dataset)/opt.model.testset.batch_size):
                # import pdb;pdb.set_trace()
                model.set_input(data)
                # print(model.name_S, model.name_F, model.name_M)
                model.test()
                images = model.get_current_visuals()
                # import pdb;pdb.set_trace()
                ssim.append(SSIM(images["fake_F"], images["real_F"], channel_axis=-1))
                psnr.append(PSNR(images["fake_F"], images["real_F"]))
                save_path = os.path.join(model.opt.save_dir, "images", str(epoch)+"_"+os.path.basename(opt.model.trainset.dataroot))
                os.makedirs(save_path, exist_ok=True)
                for key in images.keys():
                    # import pdb;pdb.set_trace()
                    save_name = os.path.join(save_path, key+"_"+os.path.basename(model.name_S[0]))
                    io.imsave(save_name, images[key])
            # import pdb;pdb.set_trace()
            mean_ssim = mean(ssim)
            mean_psnr = mean(psnr)
            output.append({"epoch": epoch, 'psnr': mean_psnr, 'ssim': mean_ssim})
            if mean_psnr > max_psnr and mean_ssim > max_ssim:
                max_psnr = mean_psnr
                max_ssim = mean_ssim
                print("max_psnr:",  max_psnr, "max_ssim:",  max_ssim)
                model.save_networks(os.path.basename(opt.model.trainset.dataroot), epoch)
                # model.save_current_images(epoch)
        # import pdb;pdb.set_trace()
        lr = model.get_current_lr()
        print("epoch:", epoch, "lr:", lr)
        model.update_learning_rate()

    with open(os.path.join(opt.model.save_dir, os.path.basename(opt.model.trainset.dataroot)+"_"+"output.json"), "w") as outfile:
        json.dump(output, outfile, indent=2, ensure_ascii=False) 
    for loss in loss_dict:
        plt.figure()
        plt.plot(loss_dict[loss])
        plt.savefig(os.path.join(opt.model.save_dir, "loss", os.path.basename(opt.model.trainset.dataroot)+"_"+loss))

if __name__ == '__main__':
    train()
    # output = [{"epoch": 1, 'psnr': 1, 'ssim': 1}, {"epoch": 1, 'psnr': 1, 'ssim': 1}, {"epoch": 1, 'psnr': 1, 'ssim': 1}, {"epoch": 1, 'psnr': 1, 'ssim': 1}, {"epoch": 1, 'psnr': 1, 'ssim': 1}]
    # with open("./output.json", "w") as outfile:
    #     json.dump(output, outfile, indent=2, ensure_ascii=False) 