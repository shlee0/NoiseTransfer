# https://github.com/zsyOAOA/DANet/blob/master/utils.py
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance

def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=3).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=3)
    sigma.clamp_(min=1e-10)

    return sigma


#########################################################################


def ks_pytorch(p_data, q_data, left_edge=-1, right_edge=1, n_bins=256):
	n = p_data.numel()
	p = torch.histc(p_data, bins=n_bins, min=left_edge, max=right_edge) / n
	q = torch.histc(q_data, bins=n_bins, min=left_edge, max=right_edge) / n

	cum_p = torch.cumsum(p, dim=0)
	cum_q = torch.cumsum(q, dim=0)
	ks_value = torch.max(torch.abs(cum_p - cum_q))
	return ks_value