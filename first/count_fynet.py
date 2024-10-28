"build with single image first - todo: find better way to determine threshold"
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from count_an_image_fynet import loadImage, particleCounting

import numpy as np
import matplotlib.pyplot as plt
import scipy

from PIL import Image

latent_dim = 512
area1 = 5  #area of particle with radius of 1 pixels
area2 = 13 #area of particle with radius of 3 pixels
threshold = 0.5 #decide if there is a particle or not

#load ground truth values of particle countings
ground_truth = scipy.io.loadmat('Desktop/URECA/dataset_AI_students/particle_counts.mat')
particle_counts = np.array(ground_truth['particle_counts'], dtype=np.int32).reshape(-1)

#load the image
input_dir = 'Desktop/URECA/first_results/FYNet'

results = np.zeros(particle_counts.shape) #result from counting algorithm
losses = np.zeros(particle_counts.shape) #the relative error between counted and ground truth

for fname in os.listdir(input_dir):
    if not fname.endswith('jpg'):
        continue
    image, idx = loadImage(input_dir, fname)
    results[idx] = particleCounting(latent_dim, image, threshold)
    losses[idx] = abs(results[idx]-particle_counts[idx]) / particle_counts[idx]

count_result = np.column_stack((particle_counts, results, losses))
np.savetxt("Desktop/URECA/first_results/output_fynet.csv", count_result, delimiter=",", fmt=("%d", "%d", "%.4f"))
