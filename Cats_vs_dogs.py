import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2

DATADIR = "C:/Users/Vivek/Pictures/PetImages"
CATEGORIES = ["DOG", "CAT"]
training_data = []
IMG_SIZE = 50
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to cats or dogs
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    return Z

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == 'relu':
        raise Exception("You haven't implemented RELU yet.")
    return A
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='sigmoid')
        caches.append(cache)
        #finish this function