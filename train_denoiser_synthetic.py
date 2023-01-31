import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# dataset path
DIV2K_TRAIN = 'train_images/DIV2K/DIV2K_train_HR'
BSDS500 = 'test_images/BSDS500/data/images'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch

import numpy as np
from PIL import Image
import datetime

from networks.ridnet_g import RIDNet_G

from models.denoising import Denoising
from datasets.denoisingDataset import DenoisingDataset
from datasets.infiniteDataLoader import InfiniteDataLoader

from utils import get_all_img_paths, save_png, make_dir
import functools



DEBUG = args.debug



PAD = 4
PATCH_SIZE = 80 + (2*PAD)
BATCH_SIZE = 32
EPOCHS = 200
STEPS_PER_EPOCH = 200 if DEBUG else 5000
NOISE_TYPE = 'pg'



# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from models.noiseTransfer import NoiseTransfer
noiser = NoiseTransfer()
for p in noiser.parameters():
	p.requires_grad = False

noiser.load('ckpts/NT')
noiser = noiser.to(device)

@torch.no_grad()
def get_sampled_noisy(gt, noisy):
	pred = noiser(gt, noisy)
	pred = torch.clamp(pred, 0.0, 1.0)
	return pred



denoiser = RIDNet_G().to(device)
WD = 1e-6
model = Denoising(denoiser, WD)





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

noise_funcs = [get_gaussian_noisy, get_poisson_noisy, get_poissonGaussian_noisy]

def get_random_noisy(x):
	r = np.random.choice(3)
	return noise_funcs[r](x)



# train dataset
rand_noise_fn = functools.partial(get_random_noisy)

tarin_gt_paths = get_all_img_paths(DIV2K_TRAIN)
if DEBUG:
	tarin_gt_paths = tarin_gt_paths[:40]

train_ds = DenoisingDataset(tarin_gt_paths, patch_size=PATCH_SIZE, noise_fn=rand_noise_fn)
train_loader = InfiniteDataLoader(train_ds, BATCH_SIZE, num_workers=2)
train_iter = iter(train_loader)



# test dataset
if NOISE_TYPE == 'gaussian':
	eval_noise_fn = functools.partial(get_gaussian_noisy, std=25)
elif NOISE_TYPE == 'poisson':
	eval_noise_fn = functools.partial(get_poisson_noisy, chi=30)
elif NOISE_TYPE == 'pg':
	eval_noise_fn = functools.partial(get_poissonGaussian_noisy, chi=30, std=25)

eval_bsds500_paths = get_all_img_paths(BSDS500)
if DEBUG:
	eval_bsds500_paths = eval_bsds500_paths[:20]
eval_bsds500_ds = DenoisingDataset(eval_bsds500_paths, noise_fn=eval_noise_fn, random_aug=False, lazy_load=True)


	





@torch.no_grad()
def evaluation(prefix_path, log_file=None):
	model.eval()

	def _save_result(eval_ds, dataset_name):
		model.reset_metrics()

		for i in range(len(eval_ds)):
			gt, noisy = eval_ds[i]
			gt = torch.unsqueeze(gt, 0).to(device)
			noisy = torch.unsqueeze(noisy, 0).to(device)
			denoised = torch.clamp(model.denoiser(noisy), 0.0, 1.0)

			logs = model.evaluate(gt, denoised)
			model.update_metrics(logs)

		eval_rets = model.result()
		print (dataset_name, file=log_file, flush=True)
		for key in eval_rets:
			print (key, eval_rets[key], file=log_file, flush=True)

	_save_result(eval_bsds500_ds, 'BSDS500')





if __name__ == '__main__':
	def _result_path(subdir):
		return os.path.join('results', 'synthetic', subdir)


	log_path = _result_path('logs')
	img_path = _result_path('imgs')
	ckpt_path = _result_path('ckpts')


	make_dir(log_path)

	log_filename = 'log'
	if DEBUG:
		log_filename += '_debug'
	log_file = open(os.path.join(log_path, '{}.txt'.format(log_filename)), mode='a')
	

	# train
	for epoch in range(1, 1+EPOCHS):
		iter_str = '{:03}'.format(epoch)
		print ('\nloop : ', iter_str, file=log_file, flush=True)
		print (datetime.datetime.now(), file=log_file, flush=True)


		model.train()
		model.reset_metrics()
		
		
		for step in range(STEPS_PER_EPOCH):
			print (epoch, step, '\r', end='')

			clean_batch, noisy_batch = train_iter.next()
			clean_batch = clean_batch.to(device)
			noisy_batch = noisy_batch.to(device)

			gen_noisy_batch = get_sampled_noisy(clean_batch, noisy_batch)

			
			if PAD > 0:
				clean_batch = clean_batch[:, :, PAD:-PAD, PAD:-PAD]
				gen_noisy_batch = gen_noisy_batch[:, :, PAD:-PAD, PAD:-PAD]
			
			logs = model.train_step(clean_batch, gen_noisy_batch, device)
			model.update_metrics(logs)


		# result
		train_rets = model.result()
		for key in train_rets:
			print (key, train_rets[key], file=log_file, flush=True)

		# save
		model.save(os.path.join(ckpt_path, iter_str))

		# evaluation
		evaluation(os.path.join(img_path, iter_str), log_file)

		# lr decay
		model.update_lr(epoch in [150,170,190])