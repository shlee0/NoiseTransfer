import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from ignite.metrics import Average

from .baseModel import BaseModel
from networks.contraE import ContraE
from networks.vunet import VUNet

from utils_gan import get_loss_adv_d, get_loss_adv_g
from utils_danet import kl_gauss_zero_center, estimate_sigma_gauss, ks_pytorch



LAMBDA_CONS = 100
LAMBDA_GAN = 100
LAMBDA_FM = 100
BLUR_KSIZE = 21




def get_contrastive_loss(logits, mask_pos, mask_all=None, temperature=0.1):
	if mask_all is None:
		mask_all = torch.ones_like(mask_pos)
	
	logits /= temperature
	
	# for numerical stability
	logits_max, _ = torch.max(logits, dim=1, keepdim=True)
	logits = logits - logits_max.detach()

	# compute log_prob
	exp_logits = torch.exp(logits) * mask_all
	eps = 1e-8
	log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)
	log_prob_pos = log_prob * mask_pos

	# compute mean of log-likelihood over positive
	positive_cnt = mask_pos.sum(1)
	mean_log_prob_pos = log_prob_pos.sum(1) / positive_cnt
	return -mean_log_prob_pos.mean()

def get_selfSup_contraloss(logits_q, logits_k, neg_queue, device):
	logits_pos = torch.mm(logits_q, logits_k.T) # [N, N]
	logits_neg = torch.mm(logits_q, neg_queue.clone().detach().to(device)) # [N, q_size]
	logits = torch.cat([logits_pos, logits_neg], dim=1) # [N, N+q_size]

	batch_size = logits.shape[0]
	mask_pos = torch.eye(batch_size, dtype=torch.float32).to(device)
	mask_neg = torch.zeros_like(logits_neg)
	mask = torch.cat([mask_pos, mask_neg], dim=1)
	contra_loss = get_contrastive_loss(logits, mask)

	# accuary
	maxk = torch.sum(mask_pos, dim=1)[0].item()
	maxk = int(maxk)
	_, indices = logits.topk(maxk, dim=1, sorted=False)
	correct = torch.gather(mask, 1, indices)
	acc = correct.mean()
	return contra_loss, acc

class Queue:
	def __init__(self, dim, q_size):
		self.K = q_size
		queue = torch.Tensor(torch.randn(dim, self.K))
		self.queue = nn.functional.normalize(queue, dim=0)
		self.queue_ptr = 0

	@torch.no_grad()
	def dequeue_and_enqueue(self, keys): # normalized_keys
		batch_size = keys.shape[0]
		ptr = self.queue_ptr
		n_push = min(self.K - ptr, batch_size) # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue[:, ptr:ptr + n_push] = keys[:n_push].T
		ptr = (ptr + n_push) % self.K  # move pointer
		self.queue_ptr = ptr


class NoiseTransfer(BaseModel):
	def __init__(self, lr=1e-4):
		super(NoiseTransfer, self).__init__('NoiseTransfer')

		n_filter = 64
		PROJ_DIM_NOISE = 128
		Q_SIZE = 4096
		BETA1, BETA2 = 0.5, 0.99
		

		NORM = 'in'
		SN = True
		self.encoder_q = ContraE(n_filter, PROJ_DIM_NOISE, norm=NORM, sn=SN, name='encoder_q')
		self.encoder_k = ContraE(n_filter, PROJ_DIM_NOISE, norm=NORM, sn=SN, name='encoder_k')
		self.generator = VUNet(n_filter, PROJ_DIM_NOISE, norm=None, sn=SN)

		self.noisy_queue = Queue(PROJ_DIM_NOISE, Q_SIZE)

		self.opt_d = optim.Adam(self.encoder_q.parameters(), lr, betas=(BETA1, BETA2), weight_decay=1e-7)
		self.opt_g = optim.Adam(self.generator.parameters(), lr, betas=(BETA1, BETA2), weight_decay=1e-7)

		self.loss_l1 = nn.L1Loss()


		self.metrics = {
			# noisy
			'loss_contra_noisy' : Average(),
			'loss_contra_noisy_adv' : Average(),
			'loss_fm_noise' : Average(),
			'contraAcc_noisy' : Average(),
			'contraAcc_noisy_adv' : Average(),

			# gan
			'loss_gan_d' : Average(),
			'loss_gan_g' : Average(),
			'loss_fm_gan' : Average(),

			'loss_consistency' : Average(),

			'akld' : Average(),
			'ks' : Average()
		}


	def forward(self, clean, ref_noisy):
		_, noisy_embeddings = self.encoder_k(ref_noisy, flag='ve', l2_norm=True)
		fake_noise = self.generator(clean, noisy_embeddings)
		return fake_noise + clean


	def train_forward(self, clean1, noisy1, noisy2, device, step):
		clean1 = clean1.to(device)
		noisy1 = noisy1.to(device)
		noisy2 = noisy2.to(device)
		real_noise1 = noisy1 - clean1
		
		if step == 'G':
			# contrastive_noise
			real_noisy_featuremap, q_noisy = self.encoder_q(noisy1, flag='ve')
			with torch.no_grad():
				_, k_noisy2 = self.encoder_k(noisy2, flag='ve')

			# generate fake image
			fake_noise1 = self.generator(clean1, k_noisy2)
			fake_noisy1 = fake_noise1 + clean1

			fake_noisy_featuremap, fake_noisy_query = self.encoder_q(fake_noisy1, flag='ve')
			loss_contra_noisy_adv, contraAcc_noisy_adv = get_selfSup_contraloss(fake_noisy_query, k_noisy2, self.noisy_queue.queue, device)
			loss_fm_noise = torch.stack([self.loss_l1(real_noisy_featuremap[i].detach(), fake_noisy_featuremap[i]) for i in range(len(real_noisy_featuremap))]).mean()

			# consistency loss
			blur_real1 = transforms.functional.gaussian_blur(noisy1, BLUR_KSIZE)
			blur_fake1 = transforms.functional.gaussian_blur(fake_noisy1, BLUR_KSIZE)
			loss_consistency = self.loss_l1(blur_real1, blur_fake1)
			
			# gan loss
			feature_real, local_logit_rq = self.encoder_q(clean1, k_noisy2, noisy1, flag='dis')
			feature_fake, local_logit_fq = self.encoder_q(clean1, k_noisy2, fake_noisy1, flag='dis')
			loss_fm_gan = torch.stack([self.loss_l1(feature_real[i].detach(), feature_fake[i]) for i in range(len(feature_real))]).mean()
			loss_gan_g = torch.stack([get_loss_adv_g(local_logit_fq[i]) for i in range(len(local_logit_fq))]).mean()
			

			# metrics
			AKLDs = Average()
			kss = Average()
			for i in range(len(real_noise1)):
				ks = ks_pytorch(real_noise1[i], fake_noise1[i])
				kss.update(ks)

				sigma_real = estimate_sigma_gauss(noisy1[i].unsqueeze(0), clean1[i].unsqueeze(0))
				sigma_fake = estimate_sigma_gauss(fake_noisy1[i].unsqueeze(0), clean1[i].unsqueeze(0))
				akld = kl_gauss_zero_center(sigma_fake, sigma_real)
				AKLDs.update(akld)


			forward_dict = {
				'loss_consistency' : loss_consistency,

				# noisy
				'loss_contra_noisy_adv' : loss_contra_noisy_adv,
				'loss_fm_noise' : loss_fm_noise,
				'contraAcc_noisy_adv' : contraAcc_noisy_adv,
				'key_noisy' : k_noisy2,

				# gan
				'loss_gan_g' : loss_gan_g,
				'loss_fm_gan' : loss_fm_gan,

				'akld' : AKLDs.compute(),
				'ks' : kss.compute(),
			}


			self.dict_temp = {
				'q_noisy' : q_noisy,
				'k_noisy2' : k_noisy2,
				'local_logit_rq' : local_logit_rq,
				'fake_noisy1' : fake_noisy1,
			}
			return forward_dict


		elif step =='D':
			q_noisy = self.dict_temp['q_noisy']
			k_noisy2 = self.dict_temp['k_noisy2']
			local_logit_rq = self.dict_temp['local_logit_rq']
			fake_noisy1 = self.dict_temp['fake_noisy1']

			loss_contra_noisy, contraAcc_noisy = get_selfSup_contraloss(q_noisy, k_noisy2, self.noisy_queue.queue, device)
			
			# loss, gan
			_, local_logit_fq = self.encoder_q(clean1, k_noisy2, fake_noisy1.detach(), flag='dis')
			loss_gan_d = torch.stack([get_loss_adv_d(local_logit_rq[i], local_logit_fq[i]) for i in range(len(local_logit_rq))]).mean()

			forward_dict = {
				# noisy
				'loss_contra_noisy' : loss_contra_noisy,
				'contraAcc_noisy' : contraAcc_noisy,

				# gan
				'loss_gan_d' : loss_gan_d,
			}
			return forward_dict


	def train_step(self, clean1, noisy1, noisy2, device):
		forward_dict = self.train_forward(clean1, noisy1, noisy2, device, 'G')

		loss_consistency = forward_dict['loss_consistency']
		loss_contra_noisy_adv = forward_dict['loss_contra_noisy_adv']
		loss_fm_noise = forward_dict['loss_fm_noise']
		loss_gan_g = forward_dict['loss_gan_g']
		loss_fm_gan = forward_dict['loss_fm_gan']
		
		
		# update G
		self.opt_g.zero_grad()

		loss_g = 0
		loss_g += loss_gan_g + LAMBDA_GAN * loss_fm_gan
		loss_g += LAMBDA_CONS * loss_consistency
		loss_g += loss_contra_noisy_adv + LAMBDA_FM * loss_fm_noise

		
		loss_g.backward()
		self.opt_g.step()


		# update D
		forward_dict_D = self.train_forward(clean1, noisy1, noisy2, device, 'D')
		loss_contra_noisy = forward_dict_D['loss_contra_noisy']
		loss_gan_d = forward_dict_D['loss_gan_d']

		self.opt_d.zero_grad()

		loss_d = loss_contra_noisy + loss_gan_d

		loss_d.backward()
		self.opt_d.step()


		# update the key encoder
		self.momentum_update_key_encoder(self.encoder_q, self.encoder_k)


		# dequeue and enqueue
		forward_dict.update(forward_dict_D)
		key_noisy = forward_dict['key_noisy']
		self.noisy_queue.dequeue_and_enqueue(key_noisy)
		

		logs = forward_dict
		for key in list(logs.keys()):
			if not key in self.metrics:
				del logs[key]
			else:
				logs[key] = logs[key].detach()
		return logs
		

	def save(self, dir_name='./', file_name=None):
		super().save(self.generator, dir_name, file_name)
		super().save(self.encoder_q, dir_name, file_name)
		super().save(self.encoder_k, dir_name, file_name)
		

	def load(self, dir_name='./', file_name=None):
		super().load(self.generator, dir_name, file_name)
		super().load(self.encoder_q, dir_name, file_name)
		super().load(self.encoder_k, dir_name, file_name)


	def set_DataParallel(self):
		self.generator = torch.nn.DataParallel(self.generator)
		self.encoder_q = torch.nn.DataParallel(self.encoder_q)
		self.encoder_k = torch.nn.DataParallel(self.encoder_k)


	@torch.no_grad()
	def init_key_encoder(self, q_net, k_net):
		# init key encoder params
		for param_q, param_k in zip(q_net.parameters(), k_net.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False  # not update by gradient


	@torch.no_grad()
	def momentum_update_key_encoder(self, q_net, k_net):
		# Momentum update of the key encoder
		m = 0.999
		for param_q, param_k in zip(q_net.parameters(), k_net.parameters()):
			param_k.data = param_k.data * m + param_q.data * (1. - m)