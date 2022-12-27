from torch import nn
import torch.nn.functional as F
from fft_conv_pytorch import fft_conv,FFTConv2d
import torch
import time

class FFT_LayerModel(nn.Module):
	def __init__(self, in_channels=1, args=None):
		super(FFT_LayerModel, self).__init__()
		#conv1
		self.conv = FFTConv2d(in_channels, args.o_channels_l1, kernel_size=args.kernel_l1, stride=1)

		self.in_linear = (args.o_channels_l1 * ((args.size - (args.kernel_l1 - 1) ) ** 2))
		self.fc = nn.Linear(self.in_linear, 10)
	
	def forward(self, x):
		bs = x.size(0)
		# torch.cuda.synchronize()
		# start = time.time()
		x = self.conv(x)
		# torch.cuda.synchronize()
		# result_time = (time.time() - start) * 1000
		x = x.view(bs, -1)
		x = self.fc(x)
		return x
		# return x, result_time

class CNN_LayerModel(nn.Module):
	def __init__(self, in_channels=1, args=None):
		super(CNN_LayerModel, self).__init__()
		#conv1
		self.conv = nn.Conv2d(in_channels, args.o_channels_l1, kernel_size=args.kernel_l1, stride=1)

		self.in_linear = (args.o_channels_l1 * ((args.size - (args.kernel_l1 - 1) ) ** 2))
		self.fc = nn.Linear(self.in_linear, 10)

 
	def forward(self, x):
		bs = x.size(0)
		# torch.cuda.synchronize()
		# start = time.time()
		x = self.conv(x)
		# torch.cuda.synchronize()
		# result_time = (time.time() - start) * 1000
		x = x.view(bs, -1)
		x = self.fc(x)
		return x
		# return x, result_time


class FFT_LayerModel_3(nn.Module):
	def __init__(self, in_channels=1, args=None):
		super(FFT_LayerModel_3, self).__init__()
		#conv1
		self.conv1 = FFTConv2d(in_channels, args.o_channels_l1, args.kernel_l1, stride=1, padding=args.padding_l1) #65 入力サイズ128/2+1が65
		self.conv2 = FFTConv2d(args.o_channels_l1, args.o_channels_l2, args.kernel_l2, stride=1, padding=args.padding_l2) #33 入力サイズ65/2+1が33
		self.conv3 = FFTConv2d(args.o_channels_l2, args.o_channels_l3, args.kernel_l3, stride=1, padding=args.padding_l3) #17

		self.in_linear = (args.o_channels_l3 * ((args.size - (args.kernel_l1 - 1) - (args.kernel_l2 - 1) - (args.kernel_l3 - 1) + (args.padding_l1 * 2) + (args.padding_l2 * 2) + (args.padding_l3 * 2)) ** 2))
		self.fc = nn.Linear(self.in_linear, 10)

	def forward(self, x):
		bs = x.size(0)
		torch.cuda.synchronize()
		start = time.time()
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		torch.cuda.synchronize()
		result_time = (time.time() - start) * 1000
		x = x.view(bs, -1)
		x = self.fc(x)
		return x, result_time

class CNN_LayerModel_3(nn.Module):
	def __init__(self, in_channels=1, args=None):
		super(CNN_LayerModel_3, self).__init__()
		#conv1
		self.conv1 = nn.Conv2d(in_channels, args.o_channels_l1, args.kernel_l1, stride=1, padding=args.padding_l1) #65 入力サイズ128/2+1が65
		self.conv2 = nn.Conv2d(args.o_channels_l1, args.o_channels_l2, args.kernel_l2, stride=1, padding=args.padding_l2) #33 入力サイズ65/2+1が33
		self.conv3 = nn.Conv2d(args.o_channels_l2, args.o_channels_l3, args.kernel_l3, stride=1, padding=args.padding_l3) #17


		self.in_linear = (args.o_channels_l3 * ((args.size - (args.kernel_l1 - 1) - (args.kernel_l2 - 1) - (args.kernel_l3 - 1) + (args.padding_l1 * 2) + (args.padding_l2 * 2) + (args.padding_l3 * 2)) ** 2))
		self.fc = nn.Linear(self.in_linear, 10)
	
	def forward(self, x):
		bs = x.size(0)
		torch.cuda.synchronize()
		start = time.time()
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		torch.cuda.synchronize()
		result_time = (time.time() - start) * 1000
		x = x.view(bs, -1)
		x = self.fc(x)
		return x, result_time


