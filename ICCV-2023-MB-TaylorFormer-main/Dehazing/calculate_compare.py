import utils
import numpy as np
from PIL import Image
import os
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn.functional as F
import math

def ssim_cv2(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def eval(image_path,gt_path,a=1):

 
    img = np.float32(utils.load_img(image_path))/255.
    print(gt_path)
    target=np.float32(utils.load_img(gt_path))/255.
    
    img = torch.from_numpy(img).permute(2,0,1)
    target=torch.from_numpy(target).permute(2,0,1)
    c,h,w=target.shape
    if a==2:
        target=target[:,:h//16*16,:w//16*16]
    input_ = img.unsqueeze(0).cuda()
    target_ = target.unsqueeze(0).cuda()

    imput_=input_.clamp_(0, 1)
    psnr_ = 10 * torch.log10(1 / F.mse_loss(input_, target_)).item()

            #down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
    ssim_ = ssim(input_,
                    target_,
                    data_range=1, size_average=False).item()


    return psnr_,ssim_

if __name__ == '__main__':

    # img_path='C:/Users/Dell/Downloads/GleNet-Paired/'
    #img_path='C:/Users/Dell/Downloads/Gle_low_psnr/'
    #img_path='D:/datasets/fivek_daylight/fivek_long_512_tif/eval/expertC_gt/'
    #img_path='/home/qiuyuwei/projects/Restormer-main_defog/Restormer-main/results/dehazing/'
    gt_path='/home/qiuyuwei/Dehamer-main/data/OTS/valid_outdoor/gt/'
    img_path='/home/qiuyuwei/Restormer-main/otss/'
    img_path2='/home/qiuyuwei/Dehamer-main/outdoor_results/'

    psnr_list=[]
    ssim_list=[]
    psnr_list2=[]
    ssim_list2=[]

    img_list = os.listdir(img_path)
    img_list.sort(key=lambda x:int(x[:4]))
    
    
    img_list2 = os.listdir(img_path2)
    img_list2.sort(key=lambda x:int(x[:4]))
    
    for i in range(500):
        index=str(i)

        
        img=None
        img=img_list[i]
        #print(img)
        #img=images
            #print(img)
        gt=img.split('_')[0]+'.png'
            #print(images)
            
        #gt_list = os.listdir(gt_path)
        #gt = None
        #for gts in gt_list:
            #if index in gts[:-1]:
                #gt = gts
        # print(img)
        # print(gt)
        if img is not None:
            a=os.path.join(img_path,img)
            b=os.path.join(gt_path,gt)
            #c=os.path.join(input_path,gt)
            psnr_cal,ssim_cal=eval(a,b)
            
            print(img,'------>psnr: ', psnr_cal,'ssim: ', ssim_cal)
            psnr_list.append(psnr_cal)
            ssim_list.append(ssim_cal)
            
            # if psnr_cal< 21:
            #     gen_path_=os.path.join(low_psnr_path,'GEN/'+img)
            #     input_path_=os.path.join(low_psnr_path,'input/'+img)
            #     gt_path_=os.path.join(low_psnr_path,'gt/'+img)
            #     img=Image.open(a)
            #     gt=Image.open(b)
            #     input=Image.open(c)
            #     img.save(gen_path_)
            #     gt.save(gt_path_)
            #     input.save(input_path_)

        else:
            continue
    t=0        
    for i in range(500):
        index=str(i)

        
        img=None
        img=img_list2[i]
        #print(img)
        #img=images
            #print(img)
        gt=img.split('_')[0]+'.png'
            #print(images)
            
        #gt_list = os.listdir(gt_path)
        #gt = None
        #for gts in gt_list:
            #if index in gts[:-1]:
                #gt = gts
        # print(img)
        # print(gt)
        
        if img is not None:
            a=os.path.join(img_path2,img)
            b=os.path.join(gt_path,gt)
            #c=os.path.join(input_path,gt)
            psnr_cal,ssim_cal=eval(a,b,2)
            print(img,'------: ', psnr_list[t]-psnr_cal,'----------', ssim_list[t]-ssim_cal)
            t=t+1
            print(img,'------>psnr: ', psnr_cal,'ssim: ', ssim_cal)
            psnr_list2.append(psnr_cal)
            ssim_list2.append(ssim_cal)
            
            # if psnr_cal< 21:
            #     gen_path_=os.path.join(low_psnr_path,'GEN/'+img)
            #     input_path_=os.path.join(low_psnr_path,'input/'+img)
            #     gt_path_=os.path.join(low_psnr_path,'gt/'+img)
            #     img=Image.open(a)
            #     gt=Image.open(b)
            #     input=Image.open(c)
            #     img.save(gen_path_)
            #     gt.save(gt_path_)
            #     input.save(input_path_)

        else:
            continue
    """
    x=range(len(psnr_list))
    plt.figure(1)
    plt.plot(x, psnr_list)
    plt.figure(2)
    plt.plot(x, ssim_list)
    plt.show()
    """
    print('final psnr: {:.2f}', np.mean(psnr_list))
    print('final ssim: {:.4f}', np.mean(ssim_list))
    print('final psnr: {:.2f}', np.mean(psnr_list2))
    print('final ssim: {:.4f}', np.mean(ssim_list2))




