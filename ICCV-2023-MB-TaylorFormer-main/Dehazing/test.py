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
parser.add_argument('--size', default='B', type=str,choices=['B','L'], help='Path to weights')
parser.add_argument('--input_dir', default='/data/QYW/ITS_SOTS/test/hazy/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data/qiuyuwei/dehazing-[4]', type=str, help='Directory for results')
parser.add_argument('--target_dir', default='/data/QYW/ITS_SOTS/test/GT/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/qiuyuwei/MB-TaylorFormer-main/Dehazing/pretrained_models/ITS-MB-TaylorFormer-B.pth'
                                         , type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
if args.size=='B':
    yaml_file = 'Dehazing/Options/MB-TaylorFormer-B.yml'
elif args.size=='L':
    yaml_file = 'Dehazing/Options/MB-TaylorFormer-L.yml'
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
model_restoration.eval()


factor = 8
datasets = ['dehazing']

for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir=args.input_dir
    target_dir=args.target_dir
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    SSIM = []
    PSNR=[]
    with torch.no_grad():
        for file_ in tqdm(files):
            img = np.float32(utils.load_img(file_))/255.

            target=np.float32(utils.load_img(os.path.join(target_dir,file_.split('/')[-1])))/255.

            img = torch.from_numpy(img).permute(2,0,1)
            target=torch.from_numpy(target).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()
            target_ = target.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]
            
            output=restored.clamp_(0, 1)
            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target_)).item()

            #down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(output,
                            target_,
                            data_range=1, size_average=False).item()

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))


            PSNR.append(psnr_val)
            SSIM.append(ssim_val)
 


        print('final PSNR:',np.mean(PSNR),'final SSIM:',np.mean(SSIM))
            