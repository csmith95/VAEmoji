import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def image_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')

class EmojiDataset(Dataset):

	'Characterizes a dataset for PyTorch'
	def __init__(self):
		'Initialization'

		max_image_num = -1
		for filename in os.listdir('../data/images'):
			img_num = int(filename[filename.index('_')+1: -4])
			max_image_num = max(img_num, max_image_num)
			
		self.num_emojis = max_image_num+1
		self.transforms = transforms.Compose([
								transforms.RandomHorizontalFlip(p=0.3), # data augmentation
								transforms.ToTensor(), # HxWxC [0, 256] -> CxHxW [0, 1]
							    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                	std=[0.229, 0.224, 0.225]) # mean/stds used by pretrained ResNet18
							])

	def __len__(self):
		'Denotes the total number of samples'
		return 256

	def __getitem__(self, index):
		'Generates one sample of data'

		x = self.transforms(image_loader('../data/images/emoji_{}.png'.format(index)))
		return x
