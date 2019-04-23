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
In this practice, two simple machine learning dataset are used including [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/), and [Caltech101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).

The pre-trained network architecture `VGG16` (trained on ImageNet) is used in this practice.

### Treat the network as a feature extractor
The network can be chopped off at an arbitrary point, normally prior to the fully-connected layers. In this practice, all the fully-connected layers are removed so that the last layer in the network is a max pooling layer, which has the output shape of 7x7x512 implying there are 512 filters of size 7x7. Therefore, we can consider these 7x7x512 = 25,088 values as a feature vector that quantifies the contents of an image. Given the feature vectors, we can train the an machine learning model, Logistic Regression classifier to recognize new classes of images.

#### Feature extraction process
The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write features into `HDF5` dataset.

The `extract_features.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/extract_features.py)) is used to extract features from an arbitrary image dataset.

After the feature extraction process, we would have a `features.hdf5` file. There are three datasets inside `features.hdf5`, including `features`, `labels`, `label_names`. The `features` dataset contains the features extracted from forward propagating the image dataset into the `VGG16` network right before the fully-connected layers. The `features` has dimension of (N x 25088), where N is total number of images in the dataset. The `labels` dataset contains the encoded labels, which has dimension of (N,). The `label_names` dataset contains the string name of the labels for the whole dataset.

#### Training a classifier on extracted features
The `train_model.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/train_model.py)) uses Logistic Regression to train a classifier on the extracted features in `features.hdf5`. We will also use grid search to find optimal value of `C`. We will finally use `pickle` to serialize Logistic Regression model to disk.

### Fine-tune networks
**This method can outperform the feature extraction method if we have sufficient data**

First we can cut off the final fully-connected layers (the head of the network) from a pre-trained convolutional neural network, `VGG16` in this practice. We then replace the head with a new set of fully-connected layers with random initializations. We freeze all the layers below the head so that their weights cannot be updated. Training data is forward propagated through the network, but the backpropagation is stopped after the fully-connected layers, which allows these layers to start to learn patterns from the highly discriminative `CONV` layers.

After fully-connected layers have started to learn patterns, we could pause the training, unfreeze the early layers (parts or all of them), and then continue the training but with small learning rate.

The `inspect_model.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/inspect_model.py)) is a helper function to check the layer name and index of every layer in a network architecture.

The `fcheadnet.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/nn/conv/fcheadnet.py)) under `pipeline/nn/conv/` directory defines fully-connected head of the network created by ourself. However, the fully-connected head is simplistic, compare to the original fully-connected head from `VGG16` which consists of two sets of 4096 `FC` layers. For most fine-tuning problems, we do not seek to replicate the original head of the network, but instead simplify it so that it is easier to fine-tune.

The `aspectawarepreprocessor.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/preprocessing/aspectawarepreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to change the size of image with respect to aspect ratio of the image.

The `imagetoarraypreprocessor.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/preprocessing/imagetoarraypreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to convert the image dataset into keras-compatile arrays.

The  `simpledatasetloader.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/pipeline/datasets/simpledatasetloader.py)) under `pipeline/datasets/` directory defines a class to load and pre-process the image dataset.

The `finetune_flowers17.py` ([check here](https://github.com/meng1994412/transfer_learning_methods/blob/master/finetune_flowers17.py)) applies fine-tuning process. It first freeze all the layers before fully-connected layers and only train the fully-connected layers for 25 epochs to "warm up" the weights of fully-connected layers. It then unfreeze the last set of `CONV` layers (3 * (`CONV`) ==> `MaxPooling`), train this set of `CONV` layers plus fully-connected layers for 50 epochs, and finally serialize the model to disk.

## Results
### Treat the network as a feature extractor
The Figure 1 shows the evaluation of the 17 Category Flower Dataset.

<img src="https://github.com/meng1994412/transfer_learning_methods/blob/master/output/feature_extraction_flower17.png" width="400">

Figure 1: Evaluation of 17 Category Flower dataset.

### Fine-tune networks
The Figure 3 shows the evaluation of just fully-connected layers, when all other layers are frozen (middle of the training). Figure 4 demonstrates the evaluation of fine-tuning the network, including training the last set of `CONV` layers and fully-connected layers.

<img src="https://github.com/meng1994412/transfer_learning_methods/blob/master/output/training_head_flower17.png" width="400">

Figure 3: Evaluation of just fully-connected layers (all other layers are frozen).

<img src="https://github.com/meng1994412/transfer_learning_methods/blob/master/output/fine_tuning_flower17.png" width="400">

Figure 4: Evaluation of fine-tuning the network.
