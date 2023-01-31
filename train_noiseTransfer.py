import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# dataset path
DIV2K_TRAIN = 'train_images/DIV2K/DIV2K_train_HR'
SIDD_MEDIUM = 'train_images/sidd/trainset/SIDD_Medium_Srgb/Data'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import numpy as np
from PIL import Image
import datetime

from models.noiseTransfer import NoiseTransfer
from datasets.syntheticNoiseDataloader import SyntheticNoiseDataloader
from datasets.SIDDTrainDataset import SIDDTrainDataset
from datasets.infiniteDataLoader import InfiniteDataLoader

from utils import get_all_img_paths, make_dir



DEBUG = args.debug



BATCH_SIZE = 32
PATCH_SIZE = 96
EPOCHS = 200
STEPS_PER_EPOCH = 100 if DEBUG else 2000



# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_cnt = torch.cuda.device_count()
model = NoiseTransfer()
model.set_DataParallel()
model = model.to(device)




# train dataset
train_div2K_paths = get_all_img_paths(DIV2K_TRAIN)
train_sidd_paths = get_all_img_paths(SIDD_MEDIUM)


train_sidd_clean_paths = []
train_sidd_noisy_paths = []

for path in train_sidd_paths:
	if 'GT' in path:
		train_sidd_clean_paths.append(path)
	else:
		train_sidd_noisy_paths.append(path)


if DEBUG:
	train_div2K_paths = train_div2K_paths[:20]
	train_sidd_clean_paths = train_sidd_clean_paths[:20]
	train_sidd_noisy_paths = train_sidd_noisy_paths[:20]


print (len(train_div2K_paths)) # 800
print (len(train_sidd_clean_paths), len(train_sidd_noisy_paths)) # 320, 320


iso_list = []
iso_imgIdx_dict = {}
for i in range(len(train_sidd_clean_paths)):
	info = os.path.dirname(train_sidd_clean_paths[i]).split('/')[-1]
	iso = int(info.split('_')[3])
	iso_list.append(iso)

	if not iso in iso_imgIdx_dict:
		iso_imgIdx_dict[iso] = []
	iso_imgIdx_dict[iso].append(i)


train_imgs_div2k = []
train_imgs_sidd_clean = []
train_imgs_sidd_noisy = []


for path in train_div2K_paths:
	train_imgs_div2k.append(np.array(Image.open(path)))

for path in train_sidd_clean_paths:
	train_imgs_sidd_clean.append(np.array(Image.open(path)))

for path in train_sidd_noisy_paths:
	train_imgs_sidd_noisy.append(np.array(Image.open(path)))




# synthetic noisy dataset
train_syn_ds = SyntheticNoiseDataloader(train_imgs_div2k, train_imgs_sidd_clean, PATCH_SIZE)

# real noisy dataset
train_sidd_ds = SIDDTrainDataset(train_imgs_sidd_clean, train_imgs_sidd_noisy, PATCH_SIZE, iso_list, iso_imgIdx_dict)

concat_ds = torch.utils.data.ConcatDataset([train_syn_ds, train_sidd_ds])
train_loader = InfiniteDataLoader(concat_ds, BATCH_SIZE, num_workers=4)
train_iter = iter(train_loader)







if __name__ == '__main__':
	def _result_path(subdir):
		return os.path.join('results', 'NT', subdir)

	log_path = _result_path('logs')
	img_path = _result_path('imgs')
	ckpt_path = _result_path('ckpts')
	make_dir(log_path)
	print (log_path)
	
	

	log_filename = 'log'
	if DEBUG:
		log_filename += '_debug'
	log_file = open(os.path.join(log_path, '{}.txt'.format(log_filename)), mode='a')

	
	# init key encoder
	model.init_key_encoder(model.encoder_q, model.encoder_k)


	# train
	for epoch in range(1, 1+EPOCHS):
		iter_str = '{:03}'.format(epoch)
		print ('\nepoch : ', iter_str, file=log_file, flush=True)
		print (datetime.datetime.now(), file=log_file, flush=True)


		model.reset_metrics()
		
		for step in range(STEPS_PER_EPOCH):
			print (epoch, step, '\r', end='')
			clean_batch, noisy_batch, noisy_batch2 = train_iter.next()
			logs = model.train_step(clean_batch, noisy_batch, noisy_batch2, device)
			model.update_metrics(logs)
			

		rets = model.result()
		for key in rets:
			print (key, rets[key], file=log_file, flush=True)

		# save
		model.save(os.path.join(ckpt_path, iter_str))