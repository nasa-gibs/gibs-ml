import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss (W, X, y, reg):
  """
  Softmax loss function (vectorized)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]

  ###############################################################################
  # Compute the loss
  ###############################################################################

  # Compute the class predictions
  scores = X.dot(W) # [N x D] x [D x C] = [N x C]
  scores -= np.max(scores,axis=1,keepdims=True) # subtract max for numerical stability 

  # Calculate the probabilities
  probabilities = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
  correct_class_probabilities = probabilities[range(num_train),y]

  # Summarize across classes that are incorrectly classified
  loss = np.sum(-np.log(correct_class_probabilities))

  # Right now the loss is a sum across all training examples but we want it 
  # to be an avaerage so we divide by num_train
  loss /= num_train

  # Add regularization to the loss
  loss += 0.5 * reg * np.sum(W*W)

  ###############################################################################
  # Compute the gradient in dW
  ###############################################################################

  # Compute the gradient
  probabilities[range(num_train),y] -= 1 # subtract 1 class for each case (a total of N) that are correctly classified
  dW = X.T.dot(probabilities)
  dW /= num_train
  
  # Add regularization to the gradient
  dW += reg * W

  return loss, dW

