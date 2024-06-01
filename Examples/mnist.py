import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Dropout, Sequential, Dense
from nn.activations import ReLu, Softmax
from nn.tensor import Tensor
from nn.utils import DataLoader, to_categorical1D, shuffle_dataset, save_object, load_object
from nn.trainer import Trainer

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], 28*28) 
    x = x.astype('float32') / 255     
    y = to_categorical1D(y, 10)       
    return shuffle_dataset(x, y)      

with np.load('Examples/data/mnist.npz', allow_pickle=True) as f:
    (train_data), (test_data) = (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])

X_train, y_train = preprocess_data(train_data[0], train_data[1])
X_test, y_test = preprocess_data(test_data[0], test_data[1]) 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

clf = Sequential([
    Dense(28*28, 256), 
    ReLu(),
    Dropout(0.3),

    Dense(256, 128), 
    ReLu(),
    Dropout(0.3),

    Dense(128, 64), 
    ReLu(),
    Dropout(0.3),

    Dense(64, 32),
    ReLu(),
    Dropout(0.3),

    Dense(32, 10),
    Softmax()
])

train_loader = DataLoader(X_train, y_train, 128, shuffle=True, autograd=True)
val_loader = DataLoader(X_val, y_val, 128, shuffle=True)
test_loader = DataLoader(X_test, y_test, 128, shuffle=True)

trainer = Trainer(clf, CCE(), Adam(clf.get_parameters(), lr=0.01))
history = trainer.train(train_loader, val_loader, 10, True)

save_object(clf, "Examples/models/mnist.model")

clf = load_object("Examples/models/mnist.model")

def plot_loss():    
    loss:dict = history[0]
    history_df = pd.DataFrame({'train': loss['train'], 'val': loss['val']})
    history_df.plot(title='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def calculate_test_accuracy():
    acc = []
    for (X, y) in test_loader():
        y_pred = clf.forward(X)
        acc.append(trainer.acc(y_pred.data, y.data))
    return np.mean(acc)

def plot_images_with_predictions():
    fig = plt.figure(figsize=(8, 5)) 
    start = 250 # see any 50 digits starting from this index
    for i in range(start, start+50): 
        plt.subplot(5, 10, (i-start)+1)  
        plt.xticks([]); plt.yticks([])
        
        img = X_test[i].reshape((28, 28))
        plt.imshow(img, cmap=plt.cm.binary) # type: ignore
        
        predicted_label = np.argmax(clf.forward(Tensor(X_test[i])).data)
        true_label = np.argmax(y_test[i])
        
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)  
    plt.show()

print(f'Test Accuracy: {calculate_test_accuracy():.4f}%') # Test Accuracy: 97.0332% 
plot_loss()
plot_images_with_predictions()