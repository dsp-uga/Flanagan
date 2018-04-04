"""
Calculates Histogram of Gradient (HOG) descriptors for every images
Applies the HOG descriptors in support vector machine
Outputs:
    1. N .npy arrays: N arrays with dimension (height, width, 100) of images in training set
    2. N .npy arrays: N arrays with dimension (height, width, 100) of hog images in training set
    3. M .npy arrays: M arrays with dimension (height, width, 100) of images in training set
    4. M .npy arrays: M arrays with dimension (height, width, 100) of hog images in training set
    5. a .dat file: trained model by specified N samples in training set
    6. M .png images: M predicted masks for M inputting testing samples
"""

import argparse
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure, io
from .visualize import visualize_figures

def load_image(hash):
    """
    Loads all 100 images in the hash folder into 100 arrays
    Generates a list of 100 images arrays
    """
    files = sorted(glob('img_data/' + hash + '/*.png'))
    images = np.array([io.imread(f) for f in files], dtype = np.int32)
    np.save("data/"+hash+".npy", images)
    return images

def calculate_hog(images, hash):
    """
    Calculates hog descriptors for every images
    Outputs a list of 100 hog images arrays
    """
    hog_images = np.array([hog(img, orientations = 8, pixels_per_cell = (4,4),
                     cells_per_block = (1,1), visualise = True)[1] for img in images],
                     dtype = np.int32)
    np.save("hog_images/" + hash + ".npy", hog_images) # save to .npy
    return hog_images

def create_train_set(N, train_list):
    """
    Creates the hog images and labesl of training set
    Outputs a hog images matrix: ((stacking pixel rows),(100 series))
    Outputs a labels vector: ((stacking pixel),)
    """
    hog_images_train = np.zeros((1, 100), dtype = np.float32)
    labels_train = np.zeros((1,), dtype = np.int32)
    i = 0
    for hash in train_list[0:N]:
        print('***** Sample ' + str(i) + ' ***********************************')
        print('***** Loading Images ******************************************')
        # images = load_image(hash)
        # images = np.array(np.load('data/'+hash+'.npy'), dtype = np.int32)
        mask_image = np.int32(io.imread('masks/' + hash + '.png'))
        mask_pixels = mask_image.flatten()
        print('***** Calculating HOG descriptors *****************************')
        # hog_images = calculate_hog(images)
        hog_images = np.load('hog_images/'+hash+'.npy')
        hog_pixels = np.array([hog_img.flatten() for hog_img in hog_images], dtype = np.float32).transpose()

        # append images and labels
        hog_images_train = np.vstack([hog_images_train, hog_pixels])
        labels_train = np.hstack([labels_train, mask_pixels])
        i += 1
    hog_images_train = np.array(hog_images_train, dtype = np.float32)[1:]
    labels_train = np.array(labels_train, dtype = np.int32)[1:]
    print('**** Training set information *************************************')
    print('hog image train type: ', type(hog_images_train))
    print('hog image train shape: ', hog_images_train.shape)
    print('labels train type: ', type(labels_train))
    print('labels train shape: ', labels_train.shape)
    print('unique labels: ', np.unique(labels_train))
    print('*******************************************************************')
    return hog_images_train, labels_train

def predict_test_set(M, test_list, model):
    """
    Predicts the labels based on the hog descriptors from the testing hashes
    Saves the labels as .png file by original image dimension
    Outputs a list of M prediction mask images
    """
    mask_pred = []
    i = 0
    for hash in test_list[0:M]:
        print('***** Sample ' + str(i) + ' ***********************************')
        print('***** Loading Images ******************************************')
        # images = load_image(hash)
        images = np.int32(np.load('data/'+hash+'.npy'))
        dim = np.array(images[0].shape)

        print('***** Calculating HOG descriptors *****************************')
        # hog_images = calculate_hog(images)
        hog_images = np.load('hog_images/'+hash+'.npy')
        hog_pixels = np.float32([hog_img.flatten() for hog_img in hog_images]).transpose()

        print('***** Predicting and Saving mask ******************************')
        mask_pred.append(save_mask(model, hash, dim, hog_pixels))
        i += 1
    return mask_pred

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
    	self.model.train(samples, cv2.ml.ROW_SAMPLE, np.int32(responses))
    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

def evaluate_model(model, samples, labels):
    """
    Compares the prediction and labels
    Outputs the accuracy and confusion matrix
    """
    pred = model.predict(samples)
    err = (labels != pred).mean()
    print('Prediction:\n',pred)
    print('Labels:\n', labels)
    print('Error:\n', labels-pred)
    print('Accuracy: %.2f %%' % ((1 - err)*100))
    confusion = np.zeros((3, 3), np.int32)
    for i, j in zip(labels, pred):
        confusion[int(i), int(j)] += 1
    print('confusion matrix: \n', confusion)
    return pred

def save_mask(model, hash, dim, hog_pixels):
    """
    Classifies cilia by trained model
    Outputs predicted array as .png file
    """
    pred = model.predict(hog_pixels)
    mask_pred = pred.reshape(dim)
    mask_path = 'preds/' + hash + '.png'
    cv2.imwrite(mask_path, mask_pred)
    return mask_pred

def main(N, M, visualized_hash, figure, show):
    """
    Main method for cilia segmentation using HOG and SVM
    """
    # Training/Testing List
    train_txt_file = open("train.txt", "r").read()
    test_txt_file = open("test.txt", "r").read()
    train_list = train_txt_file.split('\n')[0:-1]
    test_list = test_txt_file.split('\n')[0:-1]

    print('***** Training set ************************************************')
    hog_images_train, labels_train = create_train_set(N, train_list)

    if len(visualized_hash)>1:
        print('***** Visualizing figures *************************************')
        visualize_figures(visualized_hash, figure, show)

    print('***** Shuffling data **********************************************')
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(hog_images_train))
    hog_images_train, labels_train = hog_images_train[shuffle], labels_train[shuffle]

    print('***** Spliting data into training (90%) and validation set (10%) **')
    train_n = int(0.9*len(hog_images_train))
    hog_train, hog_val = np.split(hog_images_train, [train_n])
    labels_train, labels_val = np.split(labels_train, [train_n])

    print('***** Training SVM model ******************************************')
    print('hog_train dtype: ', hog_train.dtype)
    print('labels_train dtype: ', labels_train.dtype)
    model = SVM()
    model.train(hog_train, labels_train)

    print('***** Saving SVM model ********************************************')
    model.save('cilia.dat')

    print('***** Evaluating model ********************************************')
    evaluate_model(model, hog_val, labels_val)

    print('***** Testing set *************************************************')
    mask_pred = predict_test_set(M, test_list, model)
