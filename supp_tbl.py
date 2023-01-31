# %%
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


# dataset path
if 1:
	ACC_NAME = os.getlogin()
	ROOT_NAS = '/home/{}/nas'.format(ACC_NAME)
	BSDS500 = os.path.join(ROOT_NAS, 'BSDS500/data/images')
	SIDD_GT = os.path.join(ROOT_NAS, 'sidd/groundtruth')
	SIDD_NOISY = os.path.join(ROOT_NAS, 'sidd/input')
	SIDDPLUS_VALID_SRGB_GT = os.path.join(ROOT_NAS, 'siddplus/valid_srgb/gt')
	SIDDPLUS_VALID_SRGB_NOISY = os.path.join(ROOT_NAS, 'siddplus/valid_srgb/noisy')
else:
	BSDS500 = 'test_images/BSDS500/data/images'
	SIDD_GT = 'test_images/sidd/groundtruth'
	SIDD_NOISY = 'test_images/sidd/input'
	SIDDPLUS_VALID_SRGB_GT = 'test_images/siddplus/valid_srgb/gt'
	SIDDPLUS_VALID_SRGB_NOISY = 'test_images/siddplus/valid_srgb/noisy'


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import numpy as np
import functools

from PIL import Image
from utils import *
from ignite.metrics import Average

from networks.ridnet_g import RIDNet_G
from models.denoising import Denoising
from datasets.denoisingDataset import DenoisingDataset

from utils_danet import kl_gauss_zero_center, estimate_sigma_gauss, ks_pytorch




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model
from models.noiseTransfer import NoiseTransfer
noiseTransfer = NoiseTransfer().to(device)
noiseTransfer.load('ckpts/NT')

@torch.no_grad()
def get_sampled_noisy(gt, noisy):
	pred = noiseTransfer(gt, noisy)
	pred = torch.clamp(pred, 0.0, 1.0)
	return pred




# denoiser
def get_denoiser(path):
	ckpt_path = os.path.join('ckpts', path)
	denoiser = RIDNet_G()
	denoiser = denoiser.to(device)
	denoising_model = Denoising(denoiser, nn.L1Loss())
	denoising_model.load(ckpt_path)
	return denoising_model



ridnet_sidd = get_denoiser('sidd')
ridnet_synthetic = get_denoiser('synthetic')



# noise model
def get_gaussian_noisy(x, std=None):
	if std is None:
		std = np.random.uniform(0, 50)
	std /= 255
	noisy = x + np.random.normal(scale=std, size=np.shape(x))
	return np.clip(noisy, 0, 1)

def get_poisson_noisy(x, chi=None):
	if chi is None:
		chi = np.random.uniform(5, 50)
	noisy = np.random.poisson(lam=x * chi, size=np.shape(x)) / chi
	return np.clip(noisy, 0, 1)

def get_poissonGaussian_noisy(x, chi=None, std=None):
	if chi is None:
		chi = np.random.uniform(5, 50)
	if std is None:
		std = np.random.uniform(0, 50)
	std /= 255
	noisy = np.random.poisson(lam=x * chi, size=np.shape(x)) / chi
	noisy = noisy + np.random.normal(scale=std, size=np.shape(x))
	return np.clip(noisy, 0, 1)


gau_noise_fn = functools.partial(get_gaussian_noisy)
eval_gau_noise_fn = functools.partial(get_gaussian_noisy, std=25)
poisson_noise_fn = functools.partial(get_poisson_noisy)
eval_poisson_noise_fn = functools.partial(get_poisson_noisy, chi=30)
pg_noise_fn = functools.partial(get_poissonGaussian_noisy)
eval_pg_noise_fn = functools.partial(get_poissonGaussian_noisy, chi=30, std=25)











# image paths
bsds500_paths = get_all_img_paths(BSDS500)
sidd_gt_paths = get_all_img_paths(SIDD_GT)
sidd_noisy_paths = get_all_img_paths(SIDD_NOISY)
siddplus_gt_paths = get_all_img_paths(SIDDPLUS_VALID_SRGB_GT)
siddplus_noisy_paths = get_all_img_paths(SIDDPLUS_VALID_SRGB_NOISY)



# functions
@torch.no_grad()
def get_akld_ks(gt, noisy, fake_noisy): # [C,H,W]
	if gt.dim() == 3:
		gt = gt.unsqueeze(0)
		noisy = noisy.unsqueeze(0)

	real_noise = noisy - gt
	fake_noise = fake_noisy - gt
	ks = ks_pytorch(real_noise[0], fake_noise[0])

	sigma_real = estimate_sigma_gauss(noisy, gt)
	sigma_fake = estimate_sigma_gauss(fake_noisy, gt)
	alkd = kl_gauss_zero_center(sigma_fake, sigma_real)
	return alkd, ks




def get_dataset_pair(gt_paths, noisy_paths, rand_idx=None):
	if rand_idx is None:
		rand_idx = np.random.randint(len(gt_paths))
	clean = Image.open(gt_paths[rand_idx])
	noisy = Image.open(noisy_paths[rand_idx])
	clean = transforms.ToTensor()(clean).unsqueeze(0).to(device)
	noisy = transforms.ToTensor()(noisy).unsqueeze(0).to(device)
	return clean, noisy








# %%
def SIDD_PSNR_SSIM(gt_paths, noisy_paths, dataset_name):
	print ('SIDD_PSNR_SSIM')
	log_path = './supp/tbl'
	make_dir(log_path)

	log_filename = '{}_PSNR_SSIM'.format(dataset_name)
	log_file = open(os.path.join(log_path, '{}.txt'.format(log_filename)), mode='a')

	def _f(denoising_model):
		denoising_model.eval()
		denoising_model.reset_metrics()

		for i in range(len(gt_paths)):
			print (i, '\r', end='')
			gt, noisy = get_dataset_pair(gt_paths, noisy_paths, i)
			denoised = torch.clamp(denoising_model.denoiser(noisy), 0.0, 1.0)

			logs = denoising_model.evaluate(gt, denoised)
			denoising_model.update_metrics(logs)

			if 0 and i < 5:
				ret = torch.cat((gt, noisy, denoised), dim=3)
				ret = ret[0].cpu()
				save_png(ret, os.path.join(prefix_path, '{:03}_{:0.4f}_{:0.4f}'.format(i, logs['psnr'], logs['ssim'])))

		eval_rets = denoising_model.result()
		for key in eval_rets:
			print (key, eval_rets[key], file=log_file, flush=True)




	print ('noiseTransfer', file=log_file)
	_f(ridnet_sidd)














# %%
def SIDD_ALKD_KS(gt_paths, noisy_paths, dataset_name):
	print ('SIDD_ALKD_KS')
	log_path = './supp/tbl'
	make_dir(log_path)

	log_filename = '{}_ALKD_KS'.format(dataset_name)
	log_file = open(os.path.join(log_path, '{}.txt'.format(log_filename)), mode='a')

	def _f(noiser_func):
		AKLD = Average()
		KS = Average()

		for i in range(len(gt_paths)):
			
			c1, n1 = get_dataset_pair(gt_paths, noisy_paths, i)
			fake_noisy = noiser_func(c1, n1)
			akld, ks = get_akld_ks(c1, n1, fake_noisy)
			AKLD.update(akld)
			KS.update(ks)
			print (i, akld, ks, '\r', end='')

		print ('AKLD', AKLD.compute().cpu().numpy(), file=log_file, flush=True)
		print ('KS', KS.compute().cpu().numpy(), file=log_file, flush=True)


	print ('noiseTransfer', file=log_file, flush=True)
	_f(get_sampled_noisy)


















# %%
def synthetic_denoising():
	print ('synthetic_denoising')
	log_path = './supp/tbl'
	make_dir(log_path)

	log_filename = 'synthetic_denoising'
	log_file = open(os.path.join(log_path, '{}.txt'.format(log_filename)), mode='a')


	@torch.no_grad()
	def evaluation(model, eval_ds, prefix_path=None, iter=1):
		model.eval()
		model.reset_metrics()

		for _ in range(iter):
			for i in range(len(eval_ds)):
				print (i, '\r', end='')
				gt, noisy = eval_ds[i]
				gt = torch.unsqueeze(gt, 0).to(device)
				noisy = torch.unsqueeze(noisy, 0).to(device)
				denoised = torch.clamp(model.denoiser(noisy), 0.0, 1.0)

				logs = model.evaluate(gt, denoised)
				model.update_metrics(logs)

				if 0 and i < 5:
					ret = torch.cat((gt, noisy, denoised), dim=3)
					ret = ret[0].cpu()
					save_png(ret, os.path.join(prefix_path, '{:03}_{:0.4f}_{:0.4f}'.format(i, logs['psnr'], logs['ssim'])))

		eval_rets = model.result()
		for key in eval_rets:
			print (key, eval_rets[key], file=log_file, flush=True)



	print ('eval gau', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=eval_gau_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func)
		


	print ('random gau', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=gau_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func, iter=10)


	print ('eval poisson', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=eval_poisson_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func)


	print ('random poisson', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=poisson_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func, iter=10)


	print ('eval pg', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=eval_pg_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func)


	print ('random pg', file=log_file, flush=True)
	noise_func = DenoisingDataset(bsds500_paths, noise_fn=pg_noise_fn, random_aug=False, lazy_load=True)
	evaluation(ridnet_synthetic, noise_func, iter=10)






	
# %%
if __name__ == '__main__':
	# test generator
	SIDD_ALKD_KS(sidd_gt_paths, sidd_noisy_paths, 'SIDD')
	SIDD_ALKD_KS(siddplus_gt_paths, siddplus_noisy_paths, 'SIDDPlus')

	# test denosier
	SIDD_PSNR_SSIM(sidd_gt_paths, sidd_noisy_paths, 'SIDD')
	SIDD_PSNR_SSIM(siddplus_gt_paths, siddplus_noisy_paths, 'SIDDPlus')

	synthetic_denoising()