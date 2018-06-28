import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss(W, X, y, reg):
  """
  Structured SVM loss function (vectorized)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # = C
  num_train = X.shape[0] # = N

  scores = X.dot(W) # N x C   
  correct_class_scores = np.matrix(scores[np.arange(num_train), y]).T

  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[np.arange(num_train), y] = 0 # Don't count y_i
  loss = np.sum(margins)
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  binary = margins # N x C
  binary[margins > 0] = 1  
  row_sum = np.sum(binary, axis=1) # N
  binary[np.arange(num_train), y] = -row_sum.T

  dW = np.dot(X.T, binary)
  dW /= num_train

  return loss, dW
