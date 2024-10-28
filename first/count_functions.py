"build with single image first - todo: find better way to determine threshold"
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import scipy

from PIL import Image

latent_dim = 512
area1 = 5  #area of particle with radius of 1 pixels
area2 = 13 #area of particle with radius of 3 pixels
threshold = 0.5 #decide if there is a particle or not

#load data of an image
def loadImage(input_dir, fname):
    input_image = Image.open(os.path.join(input_dir, fname))
    image = np.array(input_image)[:,:,0] / 255.0
    idx_str = fname.split('_')[1].split('.')[0]
    image_index = int(idx_str) - 1
    return image, image_index

#find a cluster and return its area
def findNeighbours(image, i, j, threshold):
    """
    Calculate the area of a cluster by finding all neighbors of an '1' pixel.
    """
    area = 1
    image[i][j] = 0
    if j > 0 and image[i][j-1] >= threshold:
        area += findNeighbours(image, i, j-1, threshold)
    if j < 511 and image[i][j+1] >= threshold: #right neighbor
        area += findNeighbours(image, i, j+1, threshold)
    if i > 0 and image[i-1][j] >= threshold: #above neighbor
        area += findNeighbours(image, i-1, j, threshold)
    if i < 511 and image[i+1][j] >= threshold: #beneath neighbor
        area += findNeighbours(image, i+1, j, threshold)
    return area
        
#counting particles in the cluster
def particleCounting(latent_dim, image, threshold):
    clusters = []
    area = 0 #area of a found cluster
    
    for i in range(latent_dim):
        for j in range(latent_dim):
            if image[i][j] >= threshold:
                area = findNeighbours(image, i, j, threshold)
                clusters.append(area)
                area = 0
    
    #counting particle in a region default consider that x<13 and y<5 with x is area1, y is area2
    particles = 0 #store the counted particles
    for cluster in clusters:
        if cluster <= area1:
            particles += 1
            continue
        dif = cluster
        count = 0
        for x in range(area2):
            for y in range(area1):
                area = area1*x + area2*y
                if abs(area - cluster) < dif:
                    dif = abs(area - cluster)
                    count = x + y
                if dif == 0:
                    break
        particles += count
    return particles
