from layer import *
class NeuralNet:
  def __init__(self, input_size, layers, activation, lr=0.002, reg=0.1):
    self.layer = []
    self.input_size = input_size
    self.lr = lr
    intermediate_size = self.input_size
    for i, layer in enumerate(layers):
      self.layer.append(Dense(intermediate_size, layer, lr, reg))
      if i == len(layers)-1:
        self.layer.append(Softmax(layer))
      else:
        self.layer.append(get_activation(activation, layer))
      intermediate_size = layer

  def forward(self, X):
    input = X
    for i, layer in enumerate(self.layer):
      input = layer.forward(input)
    return input  
  
  def backward(self, y):
    dloss = y
    loss = 0
    for i in range(len(self.layer)-1, -1, -1):
      if isinstance(self.layer[i], Softmax):
        dloss, loss = self.layer[i].backward(dloss)
      else: dloss = self.layer[i].backward(dloss)
    return loss

  def fit(self, X, y, val_data, epochs=1, batch_size=256, verbose=False):
    X_val, y_val = val_data
    nbatchs = int(X.shape[0] / batch_size)
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    for j in range(epochs):  

      for i in range(nbatchs):
        self.forward(X[i*batch_size: (i+1)*batch_size])  
        self.backward(y[i*batch_size: (i+1)*batch_size])

      epoch_tacc, epoch_tloss = self.evaluate(X, y, batch_size)
      training_loss.append(epoch_tloss)
      training_accuracy.append(epoch_tacc)
      epoch_vacc, epoch_vloss = self.evaluate(X_val, y_val, batch_size)
      validation_loss.append(epoch_vloss)
      validation_accuracy.append(epoch_vacc)
      if verbose: print('Epoch (', j, ')-> Train (Loss:', round(epoch_tloss.item(), 4), ', Acc:', round(epoch_tacc.item(), 4), '), Val (Loss:', round(epoch_vloss.item(), 4), ', Acc:', round(epoch_vacc.item(), 4), ')')
    return np.array(training_loss), np.array(training_accuracy), np.array(validation_loss), np.array(validation_accuracy)

  def evaluate(self, X, y, batch_size=256):
    n = len(y)
    sum = 0
    nbatchs = int(n / batch_size)
    tloss = []
    for i in range(nbatchs):
      y_pred = self.forward(X[i*batch_size: (i+1)*batch_size]) 
      loss, _ = get_dloss(y_pred, y[i*batch_size: (i+1)*batch_size])
      y_pred = np.argmax(y_pred, axis=1) 
      sum += np.sum((y_pred == y[i*batch_size: (i+1)*batch_size].reshape(-1,)))
      tloss.append(np.mean(loss))
    loss = np.mean(np.array(tloss))
    acc = sum / n
    return acc, loss

  def predict(self, X):
    y_pred = self.forward(X)
    return np.argmax(y_pred, axis=1)
