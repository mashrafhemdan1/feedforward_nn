import cupy as np
import os
class Dense:
  def __init__(self, input_size, output_size, lr=0.002, lmda=0.1):
    # These are the weights for the input
    self.W = np.random.randn(output_size, input_size)/np.sqrt(input_size / 2)
    self.b = np.zeros((output_size, 1))
    self.dW = np.empty((output_size, input_size))
    self.db = np.empty((output_size, 1))
    self.input = np.empty((input_size, 1))
    self.lr = lr
    self.lmda = lmda
    pass
  
  def forward(self, input):
    '''
    Input: shape (input_size, 1)
    returns output: shape (output_size, 1)
    Function:
      It calculates the output of this layer

    '''
    self.input = input
    out = ((self.W @ self.input.T) + self.b).T
    return out

  def backward(self, dloss):
    '''
    Input (dloss): shape (output_size, 1)
    return output (dloss for next layer): shape (input_size, 1)
    Function:
      It calcuates the loss and the gradients
    '''
    self.dW = np.mean(dloss.reshape(dloss.shape[0], -1, 1) @ self.input.reshape(dloss.shape[0], 1, -1), axis=0) + self.lmda * self.W
    self.db = np.mean(dloss, axis=0).reshape((-1, 1)) + self.lmda * self.b
    dloss = dloss @ self.W
    
    self.gradient_descent()
    return dloss
  
  def gradient_descent(self):
    self.W = self.W - self.lr * self.dW
    self.b = self.b - self.lr * self.db

class Softmax:
  def __init__(self, size):
    self.output = np.empty((size, 1))
    self.size = size
    pass
  
  def forward(self, input):
    e_x = np.exp(input - np.max(input, axis=1).reshape((-1, 1)))
    softmax = e_x / np.sum(e_x, axis=1).reshape((-1, 1))
    self.output = softmax
    return softmax

  def backward(self, true):
    loss, dloss = get_dloss(self.output, true)
    return dloss, loss
  
def get_dloss(output, true):
  y_true_hot = np.zeros((true.size, output.shape[1]))
  y_true_hot[np.arange(true.size), true.reshape(-1,)] = 1
  dloss = output - y_true_hot
  temp = np.multiply(output, y_true_hot)
  temp = np.sum(temp, axis=1)
  loss = -np.log(temp)
  return loss, dloss
    




class ReLU:
  def __init__(self, size):
    self.output = np.empty((size, 1))
    pass
  
  def forward(self, input):
    self.output = input.clip(0)
    return self.output.copy()

  def backward(self, dloss):
    dloss = np.multiply(dloss, (self.output > 0).astype(int))
    return dloss

class Sigmoid:
  def __init__(self, size):
    self.output = np.empty((size, 1))
    pass
  
  def forward(self, input):
    self.output = 1/(1+np.exp(-input))
    return self.output.copy()

  def backward(self, dloss):
    dloss = np.multiply(dloss, self.output*(1-self.output))
    return dloss

class Tanh:
  def __init__(self, size):
    self.output = np.empty((size, 1))
    pass
  
  def forward(self, input):
    self.output = np.tanh(input)
    return self.output.copy()

  def backward(self, dloss):
    return np.multiply(dloss, 1-np.square(self.output))

class Linear:
  def __init__(self):
    pass
  
  def forward(self, input):
    return input

  def backward(self, dloss):
    return dloss

def get_activation(act, size):
  if act == 'relu':
    return ReLU(size)
  elif act == 'tanh':
    return Tanh(size)
  elif act == 'sigmoid':
    return Sigmoid(size)
  else: return Linear()