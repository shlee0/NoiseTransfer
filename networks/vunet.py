import torch
import torch.nn as nn

from .baseNet import BaseNet
from .modules import Conv2d, ConvTranspose2d, BaseModule
import functools


act_lrelu = nn.LeakyReLU()
N_RB1 = 4
N_RB2 = 3
N_RB3 = 2


class SFTLayer(BaseModule):
	def __init__(self, cond_dim, n_in_out, activation, sn, reduction_ratio=4):
		super(SFTLayer, self).__init__('SFTLayer')

		reduce_dim = (cond_dim + n_in_out) // reduction_ratio

		self.convs_gamma = nn.ModuleList([
			Conv2d(cond_dim, reduce_dim, 1, sn=sn, activation=None),
			Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
		])

		self.convs_beta = nn.ModuleList([
			Conv2d(cond_dim + 1, reduce_dim, 1, sn=sn, activation=None),
			Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
		])


	def forward(self, x, noise_z):
		b, _, h, w = x.shape
		noise_z = noise_z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

		y = noise_z
		for i in range(len(self.convs_gamma)):
			y = self.convs_gamma[i](y)
		gamma = y

		random_z = torch.randn([b, 1, h, w], device=x.device)
		y = torch.cat((noise_z, random_z), dim=1)
		for i in range(len(self.convs_beta)):
			y = self.convs_beta[i](y)
		beta = y
		
		ret = x * (1 + gamma) + beta
		return ret


class SFTResBlock(BaseModule):
	def __init__(self, n_in_out, kernel_size, cond_dim, activation=None, sn=False):
		super(SFTResBlock, self).__init__('SFTResBlock')

		self.sft1 = SFTLayer(cond_dim, n_in_out, activation=activation, sn=sn)
		self.conv1 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)
		self.sft2 = SFTLayer(cond_dim, n_in_out, activation=activation, sn=sn)
		self.conv2 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)
		
	def forward(self, x, cond):
		y = x
		y = self.sft1(y, cond)
		y = self.conv1(y)
		y = self.sft2(y, cond)
		y = self.conv2(y)
		return y + x


class VUNet(BaseNet):
	def __init__(self, filters, proj_dim_noise, norm=None, sn=False, name='VUNet'):
		super(VUNet, self).__init__(name)

		K1 = filters
		K2 = K1 * 2
		K3 = K2 * 2
		ksize = 3


		cond_dim = proj_dim_noise
		
		self.conv_first = Conv2d(3, K1, ksize, sn=sn, activation=None)

		SFTResBlock_ = functools.partial(SFTResBlock, cond_dim=cond_dim, activation=act_lrelu)

		self.blocks1 = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])

		self.conv_down1 = Conv2d(K1, K2, 4, stride=2, sn=sn, activation=None)
		

		self.blocks2 = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])

		self.conv_down2 = Conv2d(K2, K3, 4, stride=2, sn=sn, activation=None)


		self.blocks3 = nn.ModuleList([SFTResBlock_(K3, ksize, sn=sn) for _ in range(N_RB3)])

		
		self.conv_up2 = ConvTranspose2d(K3, K2, 4, stride=2, activation=None, sn=sn)

		self.conv_skip2 = Conv2d(K2 + K2, K2, ksize, sn=sn, activation=act_lrelu)
		self.blocks2_ = nn.ModuleList([SFTResBlock_(K2, ksize, sn=sn) for _ in range(N_RB2)])


		self.conv_up1 = ConvTranspose2d(K2, K1, 4, stride=2, activation=None, sn=sn)

		self.conv_skip1 = Conv2d(K1 + K1, K1, ksize, sn=sn, activation=act_lrelu)
		self.blocks1_ = nn.ModuleList([SFTResBlock_(K1, ksize, sn=sn) for _ in range(N_RB1)])
		
		self.conv_last = Conv2d(K1, 3, 1, sn=sn, activation=act_lrelu)
		
			

	def forward(self, clean, noise_z):
		inp = clean * 2 - 1 # [-1, 1]
		y = self.conv_first(inp)

		for i in range(len(self.blocks1)):
			y = self.blocks1[i](y, noise_z)
		skip1 = y

		y = self.conv_down1(y)
		for i in range(len(self.blocks2)):
			y = self.blocks2[i](y, noise_z)
		skip2 = y

		y = self.conv_down2(y)
		for i in range(len(self.blocks3)):
			y = self.blocks3[i](y, noise_z)
		y = self.conv_up2(y)

		y = self.conv_skip2(torch.cat((y, skip2), dim=1))
		for i in range(len(self.blocks2_)):
			y = self.blocks2_[i](y, noise_z)
		y = self.conv_up1(y)

		y = self.conv_skip1(torch.cat((y, skip1), dim=1))
		for i in range(len(self.blocks1_)):
			y = self.blocks1_[i](y, noise_z)

		y = self.conv_last(y)
		return y