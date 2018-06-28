from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
	"""
	Initialize the model. Weights are initialized to small random values and
	biases are initialized to zero. Weights and biases are stored in the
	variable self.params, which is a dictionary with the following keys:

	W1: First layer weights; has shape (D, H)
	b1: First layer biases; has shape (H,)
	W2: Second layer weights; has shape (H, C)
	b2: Second layer biases; has shape (C,)

	Inputs:
	- input_size: The dimension D of the input data.
	- hidden_size: The number of neurons H in the hidden layer.
	- output_size: The number of classes C.
	"""
	self.params = {}
	self.params['W1'] = std * np.random.randn(input_size, hidden_size)
	self.params['b1'] = np.zeros(hidden_size)
	self.params['W2'] = std * np.random.randn(hidden_size, output_size)
	self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
	"""
	Compute the loss and gradients for a two layer fully connected neural
	network.

	Inputs:
	- X: Input data of shape (N, D). Each X[i] is a training sample.
	- y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
	  an integer in the range 0 <= y[i] < C. This parameter is optional; if it
	  is not passed then we only return scores, and if it is passed then we
	  instead return the loss and gradients.
	- reg: Regularization strength.

	Returns:
	If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
	the score for class c on input X[i].

	If y is not None, instead return a tuple of:
	- loss: Loss (data loss and regularization loss) for this batch of training
	  samples.
	- grads: Dictionary mapping parameter names to gradients of those parameters
	  with respect to the loss function; has the same keys as self.params.
	"""
	# Unpack variables from the params dictionary
	W1, b1 = self.params['W1'], self.params['b1']
	W2, b2 = self.params['W2'], self.params['b2']
	N, D = X.shape

	# Compute the forward pass
	A1 = np.maximum(0, np.dot(X, W1) + b1) # ReLU - (N x H)
	scores = np.dot(A1, W2) + b2 # (N x C)
	
	# If the targets are not given then jump out, we're done
	if y is None:
	  return scores

	# Compute the loss
	loss = None

	# use softmax classifier loss
	scores -= np.max(scores, axis=1, keepdims=True) # subtract max for numerical stability
	probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # (N x C)

	# compute the loss: average cross-entropy loss and regularization
	correct_logprobs = -np.log(probs[range(N),y])
	data_loss = np.sum(correct_logprobs) / N
	reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
	loss = data_loss + reg_loss

	# Backward pass: compute gradients
	grads = {}

	# compute the gradient on scores
	dscores = probs
	dscores[range(N),y] -= 1
	dscores /= N

	# backpropate the gradient to the parameters
	# first backprop into parameters W2 and b2
	dW2 = np.dot(A1.T, dscores)
	db2 = np.sum(dscores.T, axis=1, keepdims=False)
	
	# next backprop into hidden layer
	dA1 = np.dot(dscores, W2.T)
	# backprop the ReLU non-linearity
	dA1[A1 <= 0] = 0
	
	# finally into W1, b1
	dW1 = np.dot(X.T, dA1)
	db1 = np.sum(dA1.T, axis=1, keepdims=False)
	
	# add regularization gradient contribution
	dW2 += reg * W2
	dW1 += reg * W1

	# check the gradients match the shape
	assert dW1.shape == W1.shape
	assert dW2.shape == W2.shape
	assert db1.shape == b1.shape
	assert db2.shape == b2.shape

	# store gradients in dict
	grads['W1'] = dW1
	grads['W2'] = dW2
	grads['b1'] = db1
	grads['b2'] = db2

	return loss, grads

  def train(self, X, y, X_val, y_val,
			learning_rate=1e-3, learning_rate_decay=0.95,
			reg=5e-6, num_iters=100,
			batch_size=200, verbose=False):
	"""
	Train this neural network using stochastic gradient descent.

	Inputs:
	- X: A numpy array of shape (N, D) giving training data.
	- y: A numpy array f shape (N,) giving training labels; y[i] = c means that
	  X[i] has label c, where 0 <= c < C.
	- X_val: A numpy array of shape (N_val, D) giving validation data.
	- y_val: A numpy array of shape (N_val,) giving validation labels.
	- learning_rate: Scalar giving learning rate for optimization.
	- learning_rate_decay: Scalar giving factor used to decay the learning rate
	  after each epoch.
	- reg: Scalar giving regularization strength.
	- num_iters: Number of steps to take when optimizing.
	- batch_size: Number of training examples to use per step.
	- verbose: boolean; if true print progress during optimization.
	"""
	num_train = X.shape[0]
	iterations_per_epoch = max(num_train / batch_size, 1)

	# Use SGD to optimize the parameters in self.model
	loss_history = []
	train_acc_history = []
	val_acc_history = []

	for it in xrange(num_iters):
	  X_batch = None
	  y_batch = None

	  # Create a random minibatch of training data and labels, storing
	  # them in X_batch and y_batch respectively.                             
	  batch_indices = np.random.choice(num_train, batch_size)
	  X_batch = X[batch_indices]
	  y_batch = y[batch_indices]

	  # Compute loss and gradients using the current minibatch
	  loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
	  loss_history.append(loss)\

	  # Update the parameters
	  self.params['W1'] -= learning_rate * grads['W1']
	  self.params['W2'] -= learning_rate * grads['W2']
	  self.params['b1'] -= learning_rate * grads['b1']
	  self.params['b2'] -= learning_rate * grads['b2']

	  if verbose and it % 100 == 0:
		print('iteration %d / %d: loss %f' % (it, num_iters, loss))

	  # Every epoch, check train and val accuracy and decay learning rate.
	  if it % iterations_per_epoch == 0:
		# Check accuracy
		train_acc = (self.predict(X_batch)[1] == y_batch).mean()
		val_acc = (self.predict(X_val)[1] == y_val).mean()
		train_acc_history.append(train_acc)
		val_acc_history.append(val_acc)

		# Decay learning rate
		learning_rate *= learning_rate_decay

	return {
	  'loss_history': loss_history,
	  'train_acc_history': train_acc_history,
	  'val_acc_history': val_acc_history,
	}

  def predict(self, X):
	"""
	Use the trained weights of this two-layer network to predict labels for
	data points. For each data point we predict scores for each of the C
	classes, and assign each data point to the class with the highest score.

	Inputs:
	- X: A numpy array of shape (N, D) giving N D-dimensional data points to
	  classify.

	Returns:
	- pred_prob: A numpy array of shape (N,C) giving the predicted probabilities
	  for each of the elements of X.
	- y_pred: A numpy array of shape (N,) giving predicted labels for each of
	  the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
	  to have class c, where 0 <= c < C.
	"""
	y_pred = None

	A1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
	scores = np.dot(A1, self.params['W2']) + self.params['b2']
	y_pred = np.argmax(scores, axis=1)

	pred_prob = scores - np.max(scores, axis=1, keepdims=True)
	pred_prob = np.exp(pred_prob) / np.sum(np.exp(pred_prob), axis=1, keepdims=True)
	
	return pred_prob, y_pred


