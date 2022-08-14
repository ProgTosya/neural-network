# Последовательная модель НС
from tensorflow.keras.models import Sequential
# Основные слои
from tensorflow.keras.layers import Dense
# Оптимизаторы
from tensorflow.keras.optimizers import Adam
# Графическое представление
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # загрузка всего датасет MNIST     
(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
x_train = x_train_org.reshape(x_train_org.shape[0], -1)   
x_test = x_test_org.reshape(x_test_org.shape[0], -1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
from tensorflow.keras import utils              
y_train = utils.to_categorical(y_train_org, 10)
y_test = utils.to_categorical(y_test_org, 10)
# Создание модели
model = Sequential()
# Добавление слоев
model.add(Dense(800, input_dim=784, activation='relu')) 
model.add(Dense(400, activation='relu')) 
model.add(Dense(10, activation='softmax'))
# Компиляция и возврат модели
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.001), 
              metrics=['accuracy'])
history = model.fit(x_train,              # Обучающая выборка
                    y_train,              # Обучающая выборка меток класса
                    batch_size=256,       # Размер батча (пакета)
                    epochs=10,           # Количество эпох обучения
                    validation_split=0.1, # доля проверочной выборки
                    verbose=1)
print(type(history))
# Отрисовка графика точности на обучающей выборке
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
# Отрисовка графика точности на проверочной выборке
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
# Отрисовка подписей осей
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
# Отрисовка легенды
plt.legend()
# Вывод графика
plt.show()