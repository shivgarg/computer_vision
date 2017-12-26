import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in np.arange(X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    loss += -np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
    dW[:,y[i]] -= X[i]
    scores = np.exp(scores)/np.sum(np.exp(scores))
    dW += np.matrix(X[i]).T*np.matrix(scores.T)

  loss /= X.shape[0]
  loss += reg*np.sum(W*W)
  dW /= X.shape[0]
  dW += 2*reg*W 


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  scores = X.dot(W)
  scores = scores.T - np.max(scores,1)
  denom = np.sum(np.exp(scores),0)
  loss = -1*np.sum(np.log(np.exp(scores[y,np.arange(X.shape[0])])/denom))
  loss /= X.shape[0]
  loss += reg*np.sum(W*W)
  scores = np.exp(scores)/denom
  scores[y,np.arange(X.shape[0])] -= 1
  dW = np.matrix(X).T*np.matrix(scores.T)
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

