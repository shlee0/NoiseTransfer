import torch
import torch.nn as nn

from .baseNet import BaseNet
from .modules import Linear, Conv2d, BaseModule


act_lrelu = nn.LeakyReLU()
N_RB1 = 2
N_RB2 = 3
N_RB3 = 4

 
class proj_head_layers(nn.Module):
	def __init__(self, n_dim, last_dim, sn=False):
		super(proj_head_layers, self).__init__()

		self.linear1 = Linear(n_dim, n_dim, sn=sn, activation=None)
		self.linear2 = Linear(n_dim, last_dim, sn=sn, activation=act_lrelu)

	def forward(self, x, l2_norm=True):
		y = self.linear1(x)
		y = self.linear2(y)
		if l2_norm:
			y = nn.functional.normalize(y, dim=1)
		return y


class ConditionalInorm(BaseModule):
	def __init__(self, cond_dim, n_in_out, activation, sn, reduction_ratio=4):
		super(ConditionalInorm, self).__init__('ConditionalInorm')

		self.inorm = nn.InstanceNorm2d(n_in_out, affine=False)

		reduce_dim = (cond_dim + n_in_out) // reduction_ratio

		self.convs_gamma = nn.ModuleList([
			Conv2d(cond_dim, reduce_dim, 1, sn=sn, activation=None),
			Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
		])

		self.convs_beta = nn.ModuleList([
			Conv2d(cond_dim, reduce_dim, 1, sn=sn, activation=None),
			Conv2d(reduce_dim, n_in_out, 1, sn=sn, activation=activation)
		])


	def forward(self, x, noise_z):
		_, _, h, w = x.shape
		noise_z = noise_z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

		y = noise_z
		for i in range(len(self.convs_gamma)):
			y = self.convs_gamma[i](y)
		gamma = y

		y = noise_z
		for i in range(len(self.convs_beta)):
			y = self.convs_beta[i](y)
		beta = y
		
		return self.inorm(x) * (1 + gamma) + beta


class SharedResBlock(BaseModule):
	def __init__(self, n_in_out, kernel_size, cond_dim, activation=None, sn=False):
		super(SharedResBlock, self).__init__('SharedResBlock')
		
		self.noise_inorm1 = nn.InstanceNorm2d(n_in_out, affine=True)
		self.gan_cin1 = ConditionalInorm(cond_dim, n_in_out, activation, sn)
		self.conv1 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)
		
		self.noise_inorm2 = nn.InstanceNorm2d(n_in_out, affine=True)
		self.gan_cin2 = ConditionalInorm(cond_dim, n_in_out, activation, sn)
		self.conv2 = Conv2d(n_in_out, n_in_out, kernel_size, activation=activation, sn=sn)
		

	def forward(self, x, flag, noise_z=None):
		y = x

		if flag == 've':
			y = self.noise_inorm1(y)
			y = self.conv1(y)
			y = self.noise_inorm2(y)
			y = self.conv2(y)

		elif flag == 'dis':
			y = self.gan_cin1(y, noise_z)
			y = self.conv1(y)
			y = self.gan_cin2(y, noise_z)
			y = self.conv2(y)

		return y + x


class ContraE(BaseNet):
	def __init__(self, filters, proj_dim_noise, norm=None, sn=False, name='ContraE'):
		super(ContraE, self).__init__(name)

		K1 = filters
		K2 = K1 * 2
		K3 = K2 * 2
		ksize = 3

		self.pre_noise = Conv2d(3, K1, ksize, sn=sn, activation=None)
		self.pre_gan = Conv2d(6, K1, ksize, sn=sn, activation=None)
		
		self.blocks1 = nn.ModuleList([SharedResBlock(K1, ksize, proj_dim_noise, sn=sn, activation=act_lrelu) for _ in range(N_RB1)])
		self.post_noise1 = nn.Sequential(*[
			Conv2d(K1, K1, 1, activation=None, sn=sn),
			Conv2d(K1, K1, ksize, activation=act_lrelu, norm=norm, sn=sn)
		])
		self.post_gan1 = nn.ModuleList([
			Conv2d(K1, K1, 1, activation=None, sn=sn),
			ConditionalInorm(proj_dim_noise, K1, act_lrelu, sn),
			Conv2d(K1, K1, ksize, activation=act_lrelu, sn=sn)
		])
		self.conv_gan1 = Conv2d(K1, 1, 1, sn=sn, activation=None)
		

		self.conv_down1 = Conv2d(K1, K2, 4, stride=2, sn=sn, activation=None)
		self.blocks2 = nn.ModuleList([SharedResBlock(K2, ksize, proj_dim_noise, sn=sn, activation=act_lrelu) for _ in range(N_RB2)])
		self.post_noise2 = nn.Sequential(*[
			Conv2d(K2, K2, 1, activation=None, sn=sn),
			Conv2d(K2, K2, ksize, activation=act_lrelu, norm=norm, sn=sn)
		])
		self.post_gan2 = nn.ModuleList([
			Conv2d(K2, K2, 1, activation=None, sn=sn),
			ConditionalInorm(proj_dim_noise, K2, act_lrelu, sn),
			Conv2d(K2, K2, ksize, activation=act_lrelu, sn=sn)
		])
		self.conv_gan2 = Conv2d(K2, 1, 1, sn=sn, activation=None)
		

		self.conv_down2 = Conv2d(K2, K3, 4, stride=2, sn=sn, activation=None)
		self.blocks3 = nn.ModuleList([SharedResBlock(K3, ksize, proj_dim_noise, sn=sn, activation=act_lrelu) for _ in range(N_RB3)])
		self.post_noise3 = nn.Sequential(*[
			Conv2d(K3, K3, 1, activation=None, sn=sn),
			Conv2d(K3, K3, ksize, activation=act_lrelu, norm=norm, sn=sn)
		])
		self.post_gan3 = nn.ModuleList([
			Conv2d(K3, K3, 1, activation=None, sn=sn),
			ConditionalInorm(proj_dim_noise, K3, act_lrelu, sn),
			Conv2d(K3, K3, ksize, activation=act_lrelu, sn=sn)
		])
		self.conv_gan3 = Conv2d(K3, 1, 1, sn=sn, activation=None)

		self.mlp_noise = proj_head_layers(K1 + K2 + K3, proj_dim_noise, sn=sn)


	def forward_ve(self, *args, l2_norm):
		x, = args
		inp = x * 2 - 1 # [-1, 1]
		y = self.pre_noise(inp)
		for i in range(len(self.blocks1)):
			y = self.blocks1[i](y, 've')
		feature1 = self.post_noise1(y)
		out1 = torch.mean(feature1, dim=(2,3))

		y = self.conv_down1(y)
		for i in range(len(self.blocks2)):
			y = self.blocks2[i](y, 've')
		feature2 = self.post_noise2(y)
		out2 = torch.mean(feature2, dim=(2,3))

		y = self.conv_down2(y)
		for i in range(len(self.blocks3)):
			y = self.blocks3[i](y, 've')
		feature3 = self.post_noise3(y)
		out3 = torch.mean(feature3, dim=(2,3))

		
		features = (feature1, feature2, feature3)
		cat = torch.cat((out1, out2, out3), dim=1)
		output = self.mlp_noise(cat, l2_norm)
		return features, output


	def forward_dis(self, *args):
		clean, noise_z, gen_noisy = args
		gen_noise = gen_noisy - clean
		clean = clean * 2 - 1 # [-1, 1]
		inp = torch.cat((clean, gen_noise), dim=1)
		

		y = self.pre_gan(inp)
		for i in range(len(self.blocks1)):
			y = self.blocks1[i](y, 'dis', noise_z)
		y = self.post_gan1[0](y)
		y = self.post_gan1[1](y, noise_z)
		feature1 = self.post_gan1[2](y)
		logit1 = self.conv_gan1(feature1)

		y = self.conv_down1(y)
		for i in range(len(self.blocks2)):
			y = self.blocks2[i](y, 'dis', noise_z)
		y = self.post_gan2[0](y)
		y = self.post_gan2[1](y, noise_z)
		feature2 = self.post_gan2[2](y)
		logit2 = self.conv_gan2(feature2)

		y = self.conv_down2(y)
		for i in range(len(self.blocks3)):
			y = self.blocks3[i](y, 'dis', noise_z)
		y = self.post_gan3[0](y)
		y = self.post_gan3[1](y, noise_z)
		feature3 = self.post_gan3[2](y)
		logit3 = self.conv_gan3(feature3)

		features = (feature1, feature2, feature3)
		logits = (logit1, logit2, logit3)
		return features, logits

	
	def forward(self, *args, flag, l2_norm=True):
		if flag == 've':
			return self.forward_ve(*args, l2_norm=l2_norm)
		elif flag == 'dis':
			return self.forward_dis(*args)
		else:
			raise NotImplementedError("flag error")