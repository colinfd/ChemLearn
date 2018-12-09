import torch
from scipy.stats import moment
import numpy as np

class PdosModel(torch.nn.Module):
	def __init__(self):
		super(PdosModel, self).__init__()
		# self.conv1 = ConvUnit(1,8)
		self.convlayers = torch.nn.Sequential(
			ConvUnit(4,16),
			ConvUnit(16,16),
			ConvUnit(16,8),
		)
		self.fc = torch.nn.Sequential(
			torch.nn.Linear(40,30),
			torch.nn.PReLU(),
			torch.nn.Linear(30,1)
		)

	def forward(self, x):
		out = self.convlayers(x)
		s = out.shape
		out = out.view(s[0], s[1]*s[2])
		out = self.fc(out)
		return out



class ConvUnit(torch.nn.Module):
	def __init__(self, inplanes=4, outplanes=4):
		super(ConvUnit, self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=4, dilation=1),
			# torch.nn.BatchNorm1d(outplanes),
			torch.nn.PReLU()
		)

	def forward(self, x):
		return self.layers(x)


class VariationalModel(torch.nn.Module):
	def __init__(self, std_dev_multiplier=1., use_moments=False, n_moments=2):
		super(VariationalModel, self).__init__()
		self.use_moments = use_moments
		self.n_moments = n_moments
		self.std_dev_multiplier = std_dev_multiplier
		self.dtype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'

		# self.conv1 = ConvUnit(1,8)
		self.convlayers = torch.nn.Sequential(
			ConvUnit(4,16),
			ConvUnit(16,16),
			ConvUnit(16,8),
		)
		self.encoder_mu = torch.nn.Linear(40, 30)
		self.encoder_logvar = torch.nn.Linear(40, 30)
		
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(30, 30),
			torch.nn.PReLU(),
			torch.nn.Linear(30,1)
		)

	def forward(self, x):
		out = self.convlayers(x)
		s = out.shape
		out = out.view(s[0], s[1]*s[2])
		mus = self.encoder_mu(out)
		representation = mus
		logvars = self.encoder_logvar(out)

		if self.training:
			representation = self.reparameterize(mus, logvars)

		if self.use_moments:
			moments = np.zeros((x.shape[0]*x.shape[1], self.n_moments))
			for i in range(self.n_moments):
				a = x.detach().cpu().numpy().reshape((x.shape[0]*x.shape[1], x.shape[2]))
				if i == 0:
					moments[:,i] = np.trapz(a, axis=1)
					print(np.where(moments[:,i]==0))
				elif i == 1:
					moments[:,i] = np.trapz((a*np.arange(a.shape[1])), axis=1)/moments[:,0]
				else:
					raise NotImplementedError('Moments above 1 not implemented')

			moments = moments.reshape((x.shape[0], x.shape[1]* self.n_moments))
			moments /= 1000.
			if moments.shape[0] != 8:
				print(np.where(np.isnan(moments)))
				asd0
			# if np.any(np.isnan(moments)):
			# 	print(moments.shape)
			# 	asd0
			# print('mean', np.mean(moments))
			# print('std', np.std(moments))
			moments = torch.from_numpy(moments).type(self.dtype)

			representation = torch.cat((representation, moments), dim=1)

		out = self.decoder(representation)

		return out, mus, logvars

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)*self.std_dev_multiplier
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)







































