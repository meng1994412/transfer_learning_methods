# Transfer Learning Practice
## Objectives
Implement two types of transfer learning when applied to deep learning for computer vision:
* Treat the network as a feature extractor.
* Replace the existing fully-connected layers with new set of fully-connected layers on top of the network, and fine-tune the weights to recognize object classes.

## Packages Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.4/) 3.4.4
* [keras](https://keras.io/) 2.2.4
* [scikit-learn](https://scikit-learn.org/stable/) 0.19.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
In this practice, three simple machine learning dataset are used including [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/), and [Caltech101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).

The pre-trained network architecture `VGG16` (trained on ImageNet) is used in this practice.

### Treat the network as a feature extractor
The network can be chopped off at an arbitrary point, normally prior to the fully-connected layers. In this practice, all the fully-connected layers are removed so that the last layer in the network is a max pooling layer, which has the output shape of 7x7x512 implying there are 512 filters of size 7x7. Therefore, we can consider these 7x7x512 = 25,088 values as a feature vector that quantifies the contents of an image. Given the feature vectors, we can train the an machine learning model, Logistic Regression classifier to recognize new classes of images.

#### Feature extraction process
The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write features into `HDF5` dataset.

The `extract_features.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/extract_features.py)) is used to extract features from an arbitrary image dataset.

After the feature extraction process, we would have a `features.hdf5` file. There are three datasets inside `features.hdf5`, including `features`, `labels`, `label_names`. The `features` dataset contains the features extracted from forward propagating the image dataset into the `VGG16` network right before the fully-connected layers. The `features` has dimension of (N x 25088), where N is total number of images in the dataset. The `labels` dataset contains the encoded labels, which has dimension of (N,). The `label_names` dataset contains the string name of the labels for the whole dataset.

#### Training a classifier on extracted features
The `train_model.py` uses Logistic Regression to train a classifier on the extracted features in `features.hdf5`. We will also use grid search to find optimal value of `C`. We will finally use `pickle` to serialize Logistic Regression model to disk.

## Results
### Treat the network as a feature extractor
The Figure 1 shows the evaluation of the 17 Category Flower Dataset.

<img src="https://github.com/meng1994412/transfer_learning_methods/blob/master/output/feature_extraction_flower17.png" width="400">

Figure 1: Evaluation of 17 Category Flower dataset.
