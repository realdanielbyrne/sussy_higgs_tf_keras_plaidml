# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 2019

@author:  Daniel Byrne

@usage:


"""

import plaidml.keras # used plaidml so I can run on any machine's video card regardless if it is NVIDIA, AMD or Intel.
import tensorflow as tf
import numpy as np
import argparse
import sklearn
import sys
import os

plaidml.keras.install_backend()
from keras.models import Sequential, Model
from keras.layers import Input,Dense, Dropout, Activation, SpatialDropout1D
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler
def GetInputDim(features = 'raw'):
  """
  The original paper variously trained on the first 22 raw features,
  7 physicist created composite features, and all the available features.

  # Arguments :
    features : raw - selects the first 22 features
              engineered - selects the next 7 features
              all - selects all the availabe features
  """
  if features == 'raw':
    start = 1
    end = 22
  elif features == 'engineered':
    start = 23
    end = 29
  else:
    start = 1
    end = 29
  return (start, end)

# Implementation of Supervised Greedy Layerwise Pre-training (Bengio et.al)
def AddLayer(model):
  """
    Adds a layer to a model, before the output layer and after all other layers.
    This is a implementation of the Supervised Greedy Layerwise Pre-training (Bengio et.al)
    alogrithm referenced in the original paper.

    # Arguments :
      model - The trained model
  """
  print("Add layer")
  hiddeninit = RandomNormal(mean=0.0, stddev=0.05, seed=None)
  output_layer = model.layers[-1]
  model.pop()

  for layer in model.layers:
    layer.trainable = False

  # add new layer and train
  model.add(Dense(300, activation='relu', kernel_initializer=hiddeninit))
  model.add(output_layer)
  return model

def BuildInitialModel(input_dim):
  """
  Builds a model with a single input layer and a binary sigmoid output layer.

  # Arguments :
    input_dim - the dimension of the 1D input vector

  """
  l1init = RandomNormal(mean=0.0, stddev=0.1, seed=None) # original model parameters for initializing input layer
  hiddeninit = RandomNormal(mean=0.0, stddev=0.05, seed=None) # original model parameters for initializing hidden layers
  outputinit = RandomNormal(mean=0.0, stddev=0.001, seed=None) # original model parameters for initializing output layer

  model = Sequential()
  model.add(Dense(300, input_dim=input_dim, activation='relu',kernel_initializer=l1init))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid',kernel_initializer=outputinit))
  return model

def CompileModel(model, lr=0.05, decay=1e-6, momentum=0.9):
  """
  Compiles the model.

  #Arguments :
    model - The untrained model
    lr - learning rate
    decay - the learning rate decay rate
    momentum - the momentum parameter

  """

  sgd = SGD(lr=lr, decay=decay, momentum = momentum)
  model.compile(loss = 'binary_crossentropy',
                optimizer = sgd,
                metrics = ['accuracy'])
  return model

def GetAllData(start,end,filepath):
  """
  Reads data from csv, extract features between start and end, and then spilt into train and test sets
  # Arguments
      start: Column of first feature
      end: Column of last feature
      filepath: path to csv
  """

  data = np.genfromtxt(filepath, delimiter=',')
  X = data[:,start:end].astype(float)


  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_train = scaler.fit_transform(X)

  y = data[:,:1].astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)
  return X_train, X_test, y_train, y_test


def score(model, X_test, y_test):
  """
  Scores the model and prints out the results.

  # Arguments :
    model - the trained model
    X_test - test set
    y_test - test labels
  """
  scores = model.evaluate(X_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])


def main():
  parser = argparse.ArgumentParser(description='Builds Higgs Boson Classifier with parameters.')

  parser.add_argument('-p','--filepath',
                      action='store',
                      type=str,
                      dest='filepath',
                      default="HIGGS_22e5.csv",
                      help="Filename of training and test data.  Column 1 should be the binary classification label.  1 - Higgs Boson, 2 - Background Process")

  parser.add_argument('-l','--learningrate',
                      action='store',
                      type=float,
                      dest='lr',
                      default=.05,
                      help="Sets the initial learning rate")

  parser.add_argument('-m','--momentum',
                      action='store',
                      type=float,
                      dest='momentum',
                      default=.9, # original model default
                      help="Sets the initial momentum")

  parser.add_argument('-d','--decay',
                      action='store',
                      type=float,
                      dest='decay',
                      default=1e-6, # original model default
                      help="Sets the learning rate decay rate.")

  parser.add_argument('-tt','--ttsplit',
                      action='store',
                      type=float,
                      dest='ttsplit',
                      default=.045, # 5e5/11e6, original model default
                      help="Train test split percentage.")

  parser.add_argument('-e','--epochs',
                      action='store',
                      type=int,
                      dest='epochs',
                      default=200, # original model default
                      help="# of epochs.")

  parser.add_argument('-b','--batch_size',
                      action='store',
                      type=int,
                      dest='batch_size',
                      default=100, # original model default
                      help="# of epochs.")

  parser.add_argument('-ft','--feature_type',
                      action='store',
                      type=str,
                      dest='feature_type',
                      default='raw',
                      help="# of features.")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'higgs_keras_plaidml_orig.h5'


  # Determine which features the model will be trained on
  start, end = GetInputDim(args.feature_type)
  input_dim  = end - start

  # Build initial model
  model = BuildInitialModel(input_dim)
  model = CompileModel(model, lr=args.lr, decay=args.decay, momentum=args.momentum)

  # Load data then train
  X_train, X_test, y_train, y_test = GetAllData(start, end, args.filepath)

  # Briefly pre-train each new layer
  n_layers = 4
  for _ in range(n_layers):
    # add layer
    AddLayer(model)
    model.fit(X_train, y_train, epochs = 7, batch_size = args.batch_size, validation_split = args.ttsplit)

  # Train the full model
  for layer in model.layers:
    layer.trainable = True

  model = CompileModel(model)

  print("Train full model")
  model.fit(X_train, y_train, epochs = args.epochs, batch_size = args.batch_size, validation_split = args.ttsplit)

  # Save model and weights
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  score(model, X_test, y_test)


main()


