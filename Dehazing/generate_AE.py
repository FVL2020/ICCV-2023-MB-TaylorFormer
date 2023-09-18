## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881



import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
from pytorch_msssim import ssim
from natsort import natsorted
from glob import glob
from basicsr.models.archs.MB_TaylorFormer import MB_TaylorFormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='/data/QYW/ITS_SOTS/test/hazy/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/home/qiuyuwei/dehazing-[4]', type=str, help='Directory for results')
parser.add_argument('--target_dir', default='/data/QYW/ITS_SOTS/test/GT/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/qiuyuwei/MB-TaylorFormer-main/Dehazing/pretrained_models/ITS-MB-TaylorFormer-B.pth'
                                         , type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = '/home/qiuyuwei/MB-TaylorFormer-main/Dehazing/Options/MB-TaylorFormer-B.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = MB_TaylorFormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.train()


factor = 8
datasets = ['dehazing']



from torchattacks.attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::


    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.MSELoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    #inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
    inp_dir=args.input_dir
    target_dir=args.target_dir
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    SSIM = []
    PSNR=[]

    for file_ in tqdm(files):
        #torch.cuda.ipc_collect()
        #torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        #print(os.path.join(target_dir,file_.split('/')[-1].split('_')[0]+'_GT.png'))
        target=np.float32(utils.load_img(os.path.join(target_dir,file_.split('/')[-1])))/255.   #.split('/')[-1].split('_')[0]+'_GT.png'

        img = torch.from_numpy(img).permute(2,0,1)
        target=torch.from_numpy(target).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        target_ = target.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h)//factor)*factor, ((w)//factor)*factor
        input_=input_[:,:,:H,:W]
        target_=target_[:,:,:H,:W]
        #restored = model_restoration(input_)

        # Unpad images to original dimensions
        #restored = restored[:,:,:h,:w]

        #restored=restored.clamp_(0, 1)
        attack2 = PGD(model_restoration, eps=1/255, alpha=1/1020, steps=10, random_start=True)
        adv_images = attack2(input_, target_)

        #restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        adv_images = torch.clamp(adv_images,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(adv_images))


            