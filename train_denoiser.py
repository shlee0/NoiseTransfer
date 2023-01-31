import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# dataset path
SIDD_MEDIUM = 'train_images/sidd/trainset/SIDD_Medium_Srgb/Data'
SIDD_GT = 'test_images/sidd/groundtruth'
SIDD_NOISY = 'test_images/sidd/input'
SIDDPLUS_VALID_SRGB_GT = 'test_images/siddplus/valid_srgb/gt'
SIDDPLUS_VALID_SRGB_NOISY = 'test_images/siddplus/valid_srgb/noisy'


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

from datasets.SIDDTrainDataset import SIDDTrainDataset

from utils import get_all_img_paths, save_png, make_dir


DEBUG = args.debug


PAD = 4
PATCH_SIZE = 80 + (2*PAD)
BATCH_SIZE = 32
EPOCHS = 200
STEPS_PER_EPOCH = 200 if DEBUG else 5000



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
WD = 1e-5
model = Denoising(denoiser, WD)




# train dataset
sidd_train_paths = get_all_img_paths(SIDD_MEDIUM)
sidd_train_gt_paths = []
sidd_train_noisy_paths = []
if DEBUG:
	sidd_train_paths = sidd_train_paths[:20]


for i in range(len(sidd_train_paths)):
	path = sidd_train_paths[i]
	if 'GT' in path:
		sidd_train_gt_paths.append(path)
	else:
		sidd_train_noisy_paths.append(path)

print (len(sidd_train_gt_paths), len(sidd_train_noisy_paths))

clean_sidd_imgs = []
noisy_sidd_imgs = []
clean_div2k_imgs = []


for i in range(0, len(sidd_train_gt_paths)):
	clean_sidd_imgs.append(np.array(Image.open(sidd_train_gt_paths[i])))
	noisy_sidd_imgs.append(np.array(Image.open(sidd_train_noisy_paths[i])))



iso_list = []
iso_imgIdx_dict = {}
for i in range(len(sidd_train_gt_paths)):
	info = os.path.dirname(sidd_train_gt_paths[i]).split('/')[-1]
	iso = int(info.split('_')[3])
	iso_list.append(iso)

	if not iso in iso_imgIdx_dict:
		iso_imgIdx_dict[iso] = []
	iso_imgIdx_dict[iso].append(i)


train_sidd_ds = SIDDTrainDataset(clean_sidd_imgs, noisy_sidd_imgs, PATCH_SIZE, iso_list, iso_imgIdx_dict)
train_sidd_loader = InfiniteDataLoader(train_sidd_ds, BATCH_SIZE, 2)
train_sidd_iter = iter(train_sidd_loader)




# test dataset
eval_sidd_gt_paths = get_all_img_paths(SIDD_GT)
eval_sidd_noisy_paths = get_all_img_paths(SIDD_NOISY)
eval_siddplus_gt_paths = get_all_img_paths(SIDDPLUS_VALID_SRGB_GT)
eval_siddplus_noisy_paths = get_all_img_paths(SIDDPLUS_VALID_SRGB_NOISY)
if DEBUG:
	eval_sidd_gt_paths = eval_sidd_gt_paths[:20]
	eval_sidd_noisy_paths = eval_sidd_noisy_paths[:20]
	eval_siddplus_gt_paths = eval_siddplus_gt_paths[:20]
	eval_siddplus_noisy_paths = eval_siddplus_noisy_paths[:20]
eval_sidd_ds = DenoisingDataset(eval_sidd_gt_paths, eval_sidd_noisy_paths, random_aug=False, lazy_load=True)
eval_siddplus_ds = DenoisingDataset(eval_siddplus_gt_paths, eval_siddplus_noisy_paths, random_aug=False, lazy_load=True)




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

	_save_result(eval_sidd_ds, 'SIDD')
	_save_result(eval_siddplus_ds, 'SIDD+')








if __name__ == '__main__':
	def _result_path(subdir):
		return os.path.join('results', 'SIDD', subdir)


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

			clean_batch, noisy_batch, noisy_batch2 = train_sidd_iter.next()
				
			clean_batch = clean_batch.to(device)
			noisy_batch = noisy_batch.to(device)
			noisy_batch2 = noisy_batch2.to(device)

			# gen_noisy_batch = get_sampled_noisy(clean_batch, noisy_batch) # paired
			gen_noisy_batch = get_sampled_noisy(clean_batch, noisy_batch2) # unpaired


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