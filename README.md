# Cilia Segmentation

This repository contains various algorithms implemented on cilia images segmentation which are completed on CSCI 8360, Data Science Practicum at the University of Georgia, Spring 2018.

This project uses the time series grayscale 8-bit images of cilia biopsies taken with DIC optics published in the study [Automated identification of abnormal respiratory ciliary motion in nasal biopsies](http://stm.sciencemag.org/content/7/299/299ra124). There are 325 videos, 211 (65%) for training set and 114 (35%) for testing set. For each video, they are formed by 100 successive frames and performed as a 0.5 seconds real-time video. 3-label masks come with the videos and the pixels of them are colored according to the corresponding videos as **2** for cilia, **1** for a cell, and **0** for the background. In this case, we are only interested in cilia segmentation, the misclassification of the cells and backgrounds are not considered here.

In this repository, we are offering three different methods as follows using different packages to locate the cilia and segment them out from the surrounding images.

1. Optical Flow using OpenCV
2. Convolutional Neural Network using tf-unet
3. Histogram of Gradient using scikit-image and Support Vector machine using OpenCV

Read more details about each algorithm and their applications in our [WIKI](https://github.com/dsp-uga/Flanagan/wiki) tab.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/)

### Environment Setting

  1. **Clone this repository**
  ```
  $ git clone https://github.com/dsp-uga/Flanagan
  $ cd Flanagan
  ```

  2. **Create conda environment based on `environment.yml` offers in this repository**
  ```
  $ conda env create -f environments.yml -n cilia python=3.6
  $ source activate cilia
  ```

  3. **Install Unet**
  ```
  $ git clone https://github.com/jakeret/tf_unet
  $ cd tf_unet
  $ python setup.py install
  $ rm -rf tf_unet
  ```

## Running the tests

```
python -m [algorithm] [args-for-the-algorithm]
```

##### Algorithms

  - `OpticalFlow`: Running Optical Flow
  - `CNN`: Running Convolutional Neural Network
  - `HOG2SVM`: Running Support Vector Machine by HOG images

Each folders includes one module and you can run it through the command above. Each module provides their own arguments. Use `help()` to know more details when running the algorithms.


## Evaluation

The results count on the ratio of the values of intersection over union. Take the **intersection** as the area of overlap between the predicted region and the actual region, and the **union** as the area of the union of the predicted region and the actual region.

<p align = "center">
<img src = "https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png" width = 250>
</p>

## Test Results


| Module    | arguments             | Mean IoU     |
|-----------|-----------------------|--------------|
|OpticalFlow|
|CNN        |
|HOG+SVM    |N=30                   | 9.01251      |


## Discussion

  1. **Optical Flow**

      -
      -
      -

  2. **Convolutional Neural Network**

      -
      -
      -

  3. **Support Vector Machine with Histogram of Gradient**

      - HOG works well in detecting the shape of cells but not cilia (Cilia are too thin and small to detect)
      - Instead of inputting HOG feature descriptors in SVM, we used HOG images since the labels are assigned to each pixel but not each image. That is to say, stacking all pixels of 211 training videos, there will be 4 million instances in SVM.
      - Takes forever to run the model because of the great amount of instances
      - Visualization of one frame of the video, one hog frame of the video, and the mask:

      <p align = "center">
      <img src = "img/hog2svm_visualization.png" width = >
      </p>


## Authors
(Ordered alphabetically)

- **I-Huei Ho** - [melanieihuei](https://github.com/melanieihuei)
- **Nicholas Klepp** - [NBKlepp](https://github.com/NBKlepp)
- **Vinay Kumar** - [vinayawsm](https://github.com/vinayawsm)

See the [CONTRIBUTORS](CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
