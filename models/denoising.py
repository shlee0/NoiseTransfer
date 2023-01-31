import os
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.metrics import Average
from piq import psnr, ssim

from .baseModel import BaseModel

class Denoising(BaseModel):
	def __init__(self, denoiser, wd, lr=1e-4):
		super(Denoising, self).__init__('Denoising')

		self.denoiser = denoiser
		self.criterion = nn.L1Loss()
		self.optimizer = optim.Adam(denoiser.parameters(), lr, weight_decay=wd)
		self.metrics = {
			'loss' : Average(),
			'psnr' : Average(),
			'ssim' : Average()
		}


	# @torch.no_grad()
	def evaluate(self, clean, denoised):
		loss = self.criterion(denoised, clean)
		denoised = torch.clamp(denoised, 0.0, 1.0)
		psnr_val = psnr(clean, denoised)
		ssim_val = ssim(clean, denoised)

		logs = {
			'loss' : loss,
			'psnr' : psnr_val,
			'ssim' : ssim_val
		}
		return logs


	def train_step(self, clean, noisy, device):
		clean = clean.to(device)
		noisy = noisy.to(device)
		self.optimizer.zero_grad()
		denoised = self.denoiser(noisy)

		logs = self.evaluate(clean, denoised)
		loss = logs['loss']
		loss.backward()
		self.optimizer.step()

		return logs


	def save(self, dir_name='./', file_name=None):
		super().save(self.denoiser, dir_name, file_name)


	def load(self, dir_name='./', file_name=None):
		super().load(self.denoiser, dir_name, file_name)


	def update_lr(self, decay):
		if decay:
			self.optimizer.param_groups[0]['lr'] *= 0.5