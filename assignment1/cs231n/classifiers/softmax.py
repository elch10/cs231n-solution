import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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

  num_images = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_images):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    softmax = exp_scores / sum_exp_scores

    loss += -np.log(softmax[y[i]])

    temp = X[i, :, np.newaxis].dot(softmax[np.newaxis, :])
    temp[:, y[i]] = (softmax[y[i]] - 1) * X[i]
    dW += temp
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss = loss / num_images + reg * np.sum(W * W)
  dW = dW / num_images + 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_images = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1)
  softmax = exp_scores / sum_exp_scores[:, None]

  loss = np.sum(-np.log(softmax[np.arange(num_images), y]))

  for i in range(num_images): # It is so hard to vectorize...
    temp = X[i, :, None].dot(softmax[i, None, :])
    temp[:, y[i]] = (softmax[i, y[i]] - 1) * X[i]
    dW += temp
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss = loss / num_images + reg * np.sum(W * W)
  dW = dW / num_images + 2 * reg * W

  return loss, dW

