import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

def make_dir(dir_path):
	if not(os.path.isdir(dir_path)):
		os.makedirs(dir_path)
	
def get_all_img_paths(path_root, min_h=0, min_w=0):
	paths = []

	for (dirpath, dirnames, filenames) in os.walk(path_root):
		filenames = [f for f in filenames if not f[0] == '.']
		dirnames[:] = [d for d in dirnames if not d[0] == '.']

		for file in filenames:
			if (file.lower().endswith(tuple(['.bmp', '.jpg', '.png']))):
				path = os.path.join(dirpath, file)
				if min_h == 0 and min_w == 0:
					paths.append(path)
				else:
					img = Image.open(path)
					if(img.mode != 'RGB'):
						continue
					img_size = img.size
					h = img_size[1]
					w = img_size[0]
					if(min_h <= h and min_w <= w):
						paths.append(path)
	return sorted(paths)

def get_folder_img_name(img_path):
	dir_name = os.path.basename(os.path.dirname(img_path))
	img_name = os.path.splitext(os.path.basename(img_path))[0]
	return dir_name, img_name

def save_png(pilImg, path):
	if type(pilImg) != Image.Image:
		pilImg = transforms.ToPILImage()(pilImg)
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	make_dir(dir_path)
	pilImg.save(path + '.png')

def randInt(x, size):
	return np.random.randint(x - size + 1)
	
def randPos(h, w, hsize, wsize=None):
	if wsize is None:
		wsize = hsize
	y = randInt(h, hsize)
	x = randInt(w, wsize)
	return y, x

def np_crop(nparr, y, x, h, w=None):
	if w == None:
		w = h
	return nparr[y:y+h, x:x+w]

def downscale_args(scale, *args):
	return np.array(args) // scale

def upscale_args(scale, *args):
	return np.array(args) * scale

def modular_crop(img_np, scale):
	h, w, c = img_np.shape
	h, w = downscale_args(scale, h, w)
	h, w = upscale_args(scale, h, w)
	return np_crop(img_np, 0, 0, h, w)

def color_aug(np_img):
	perm = np.arange(3)
	np.random.shuffle(perm)
	return np_img[:,:,perm]

def minmax_norm(img):
	if type(img) == torch.Tensor:
		mini = torch.min(img)
		maxi = torch.max(img)
	else:
		mini = np.min(img)
		maxi = np.max(img)
	return (img - mini) / (maxi - mini)