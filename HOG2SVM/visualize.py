"""
Visualize original image, hog image, and mask together
"""

import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, io
# from .hog2svm import load_image

def visualize_hog(mask, images, hog_images, show):
    """
    Visualizes the mask, images and hog images
    Generates 100 figures, each contains three images
    """
    for i in range(100):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12,4), sharex = True, sharey = True)
        # Original Image
        image = images[i]
        ax1.axis('off')
        ax1.imshow(image, cmap = plt.cm.gray)
        ax1.set_adjustable('box-forced')
        # HoG Image
        hog_image = hog_images[i]
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range = (0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap = plt.cm.gray)
        ax2.set_adjustable('box-forced')
        # Mask Image
        ax3.axis('off')
        ax3.imshow(mask, cmap = plt.cm.gray)
        ax3.set_adjustable('box-forced')
        # Save figure
        plt.savefig('figures/hog/'+ hash + '_comp_' + str(i) +'.png')
        # Show figure
        if show == True: plt.show()

def visualize_pred(mask, pred, show):
    """
    Visualizes the mask and predicted mask
    Generates a figure, each with two images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8,4), sharex = True, sharey = True)
    # Original mask
    ax1.axis('off')
    ax1.imshow(mask, cmap = plt.cm.gray)
    ax1.set_adjustable('box-forced')
    # Predicted mask
    ax2.axis('off')
    ax2.imshow(pred, cmap = plt.cm.gray)
    ax2.set_adjustable('box-forced')
    # Save figure
    plt.savefig('figures/pred/'+ hash + '_pred.png')
    # Show figure
    if show == True: plt.show()

def visualize_figures(h, f, v):
    """
    Visualizes figures according to the options
    Figures are showed based on the options
    """
    images = np.int32(np.load('data/'+h+'.npy'))
    mask = io.imread('masks/' + h + '.png')
    hog_images = np.load('hog_images/' + h + '.npy')
    pred = io.imread('preds/' + h + '.png')
    # Visualize figures
    if f == 'hog':
        visualize_hog(mask, images, hog_images, v)
    if f == 'pred':
        visualize_pred(mask, pred, v)
