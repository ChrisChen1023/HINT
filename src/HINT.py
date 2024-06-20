import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import wandb
import lpips
import torchvision
import time

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class HINT():
    def __init__(self, config):
        self.config = config


        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        #train mode
        if self.config.MODE == 1:

            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)

        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                print('model == 2')
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):

        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def train(self):
        wandb.watch(self.inpaint_model, self.psnr, log='all', log_freq=10)
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            for items in train_loader:

                self.inpaint_model.train()
                if model == 2:
                    images, masks = self.cuda(*items)
                # inpaint model

                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss= self.inpaint_model.process(images,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0:
                        wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                   'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                   'dis_loss': dis_loss}, step=iteration)

                ###################### visialization
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )


                    path_masked = os.path.join(self.results_path,self.model_name,'masked')
                    path_result = os.path.join(self.results_path, self.model_name,'result')
                    path_joint = os.path.join(self.results_path,self.model_name,'joint')
                    name = self.train_dataset.load_name(epoch-1)[:-4]+'.png'

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)

                    masked_images = self.postprocess(images*(1-masks)+masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint,name[:-4]+'.png'))

                    images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                    imsave(masked_images,os.path.join(path_masked,name))
                    imsave(images_result,os.path.join(path_result,name))

                    print(name + ' complete!')
                    
                ##############

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')


    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)
        cal_mean_nme = self.cal_mean_nme()

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        
        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []
        
        print('here')
        index = 0
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            # inpaint model
            if model == 2:


                inputs = (images * (1 - masks))
                with torch.no_grad():
                    tsince = int(round(time.time()*1000))
                    outputs_img = self.inpaint_model(images, masks)
                    ttime_elapsed = int(round(time.time()*1000))-tsince
                    print('test time elaspsed {}ms'.format(ttime_elapsed))
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
                
                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)                
                
                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)

                print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                                ssim, np.average(ssim_list),
                                                                                l1_loss, np.average(l1_list),
                                                                                pl, np.average(lpips_list),
                                                                                len(ssim_list)))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked4060')
                path_result = os.path.join(self.results_path, self.model_name,'result4060')
                path_joint = os.path.join(self.results_path,self.model_name,'joint4060')


                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint,name[:-4]+'.png'))

                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                print(name + ' complete!')

            # inpaint with joint model
        torch.onnx.export(model, images_joint, 'model.onnx')
        wandb.save('model.onnx')
        print('\nEnd Testing')
        
        print('edge_psnr_ave:{} edge_ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
                                                                                 np.average(ssim_list),
                                                                                 np.average(l1_list),
                                                                                 np.average(lpips_list)))




    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim
    
    class cal_mean_nme():
        sum = 0
        amount = 0
        mean_nme = 0

        def __call__(self, nme):
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            return self.mean_nme

