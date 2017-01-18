from scipy import ndimage
import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt
from numpy import genfromtxt
from itertools import chain

'''
This program converts the mnist dataset saved in the files 'traininset_features.csv' and 'trainingset_labels.csv'
into python lists and arrays.

The trainingset is converted into a list of 2-d numpy arrays, while the labels are converted to a list of integers

The mnist dataset can be found at http://yann.lecun.com/exdb/mnist/ and images at www.cs.nyu.edu/~roweis/data/ 
The data has been preprocessed from a csv converted version (http://pjreddie.com/projects/mnist-in-csv/)
'''
####--------------function definitions-----------------------

## writes a list of 2d feature arrays into the file 'filename' (i.e., myfeatures.txt)  
def write_features_to_file(features, filename):
    data_flattened = [list(chain.from_iterable(item)) for item in features]
    f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            filename), 'w')
    np.savetxt(f ,data_flattened, delimiter=",", fmt='%03f', newline = '\n')
    return 0

## writes a list of labels into the file 'filename' (i.e., mylabels.txt)  
def write_labels_to_file(labels, filename):
    f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            filename), 'w')
    np.savetxt(f ,labels, delimiter=",", fmt='%i')
    return 0

## loads the features into a list of 2d numpy arrays
def load_features(filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            filename)
    features = np.loadtxt(path, dtype=float, delimiter =',')
    dim = int(np.sqrt(len(features[0])))   
    features = [np.array(item).reshape((dim,-1)) for item in features]
    return features

## loads the labels into a list
def load_labels(filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            filename)
    labels = np.loadtxt(path, dtype=int, delimiter =',')
    return labels

## generates an image of the feature vector
def show_feature_vector(twodarray):
    plt.imshow(twodarray, cmap='Greys', interpolation='nearest')
    plt.show()
    return 0

def convert_to_grayscale(features):
    grayscale_features=[item/255. for item in features]
    return grayscale_features

def convert_to_blackwhite(features, threshold):
    bw_features = features
    for item in features:
        for i in range(len(item)):
            for j in range(len(item[0])):
                if item[i,j] > threshold:
                    item[i,j] = 1
                else:
                    item[i,j] = 0
    return bw_features

##---------------END FUNCTIONS--------------------------


## read in trainingset features (inputs)
trainingset_features = load_features("trainingset_features.csv")
##convert original data to gray scale (renormalise numbers to interval [0,1])
trainingset_features = convert_to_grayscale(trainingset_features )
##convert to black-white image (binarise numbers to 0 or 1)
trainingset_features = convert_to_blackwhite(trainingset_features, 0.5)


## read in trainingset labels (target outputs)
trainingset_labels = load_labels("trainingset_labels.csv")


##SOME THINGS YOU CAN DO

## show an image of the i'th feature vector in the training set 
show_feature_vector(trainingset_features[8768])

## print the feature array of the i'th feature vector in the training set
print trainingset_features[8768] 

## print the label of the i'th feature vector in the training set
print trainingset_labels[8768] 

#### to do something with every entry of the feature array:
## for item in features:
##      for i in range(len(item)):
##          for j in range(len(item[0])):
##              do something with item[i,j]


#### to save a list of labels called 'mylabels.txt'
##write_labels_to_file(trainingset_labels,"mylabels.txt")
