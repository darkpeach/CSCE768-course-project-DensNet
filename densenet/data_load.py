import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Protein_Dataset(Dataset):
	def __init__(self, start, end):
		self.cb6133 = np.load('../datasets.npy')
		self.start = start
		self.end = end
		self.data = self.cb6133[self.start: self.end, :, range(42)]
		self.label = self.cb6133[self.start: self.end, :, range(42, 51)]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		features = np.transpose(self.data[idx])
		labels = np.transpose(self.label[idx])
		return torch.from_numpy(features).float(), torch.from_numpy(labels).float()
