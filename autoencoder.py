import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense ,Input
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


def plot_autoencoder_outputs(model, n, dims):
    decoded_imgs = model.predict(test)

    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Original Images')

        # plot reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Reconstructed Images')
    plt.show()

train=pd.read_csv(r'C:\Users\HP\Desktop\mnist_digits\mnist_train.csv')
test=pd.read_csv(r'C:\Users\HP\Desktop\mnist_digits\mnist_test.csv')
print(type(train),train.shape)
train = np.array(train)
test = np.array(test)


train=train[:,1:]
test=test[:,1:]

train = train.astype('float32') / 255.0
test = test.astype('float32') / 255.0
print(type(train),train.shape)

input_size=784
hidden_layer=128
hidden_layer2=64
code=16
model=Sequential()
model.add(Dense(hidden_layer,input_shape=(input_size,),activation='relu'))
model.add(Dense(hidden_layer2,activation='relu'))
model.add(Dense(code,activation='relu'))
model.add(Dense(hidden_layer2,activation='relu'))
model.add(Dense(hidden_layer,activation='relu'))
model.add(Dense(input_size,activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer=Adadelta(),metrics=['accuracy'])
model.fit(train,train,batch_size=64,epochs=30,validation_split=0.33)


plot_autoencoder_outputs(model,5,(28,28))