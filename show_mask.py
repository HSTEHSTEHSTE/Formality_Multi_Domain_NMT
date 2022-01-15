import pickle
import torch
import os
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

mask_file = open("C://Users//HSTE//JHU//Fall 2021//Machine Translation//Formality_Multi_Domain_NMT//mask.pt", mode="rb")
mask = torch.load(mask_file).to(device='cpu')
mask = mask[1] - mask[0]
vmax = torch.max(mask)
vmin = torch.min(mask)
plt.imshow(mask, aspect="auto", vmax=vmax, vmin=vmin)
plt.show()