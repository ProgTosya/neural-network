from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
class_count = 10
model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(class_count, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
x_train = x_train_org.reshape(x_train_org.shape[0], -1)
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = utils.to_categorical(y_train_org, class_count)
y_test = utils.to_categorical(y_test_org, class_count)
utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)
model.save_weights('model.h5')
model.load_weights('model.h5')
n_rec = np.random.randint(x_test_org.shape[0])
plt.imshow(x_test_org[n_rec], cmap='gray')
plt.show()
x = x_test[n_rec]
print(x.shape)
x=np.expand_dims(x, axis=0)
print(x.shape)
prediction = model.predict(x)
print(prediction)
pred = np.argmax(prediction)
print(f'Наибольшая вероятность что это число {pred}')

