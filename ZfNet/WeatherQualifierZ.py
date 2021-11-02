from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import h5py
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import random
import pickle
import cv2
import os
import numpy as np

# создание парсера аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
    help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# инициализация данных, меток
print("[INFO] loading images...")
data = []
labels = []
 
# пути к изображениям. Shuffle
print('[INFO] shuffle images')
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
 
# цикл по изображениям
print('[INFO] preprocessing images')
for imagePath in imagePaths:
    # загрузка изображений, resize, сглаживание
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (227, 227))
    data.append(image)
 
    # извлечение метки класса из пути к изображению, обновление списка меток
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
# масштабирование интенсивности пикселей в диапазон [0, 1]
print('[INFO] intensity changing')
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# создание обучающейся и тестовой выборок. 80% - обучение, 20% - тест
print('[INFO] making test and training images')
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

# конвертирование меток из целых чисел в векторы
print('[INFO] converting labels images')
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print('[INFO] done')

# создание модели нейронной сети

np.random.seed(1000)

model = Sequential()

model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(7, 7),
                 strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(4, 4)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                    padding='same'))
model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                 padding='valid'))
model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1),
                 padding='valid'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                 padding='valid'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                    padding='same'))

model.add(Flatten())

model.add(Dense(4096, input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(6))
model.add(Activation('softmax'))


# Компиляция модели
model.compile(loss=losses.categorical_crossentropy,
              optimizer='SGD', metrics=["accuracy"])

# Fitting
EPOCH = 1
es = EarlyStopping(monitor='val_loss',
                   mode='min', verbose=EPOCH, patience=10)
mcp_save = ModelCheckpoint('.mdl_wts.gdf5',
                           save_best_only=True, monitor='val_loss', mode='min')
rll = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=100, batch_size=8, callbacks=[es, mcp_save, rll]) 

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=8)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

 # сохранение модели и бинаризатора меток на диск
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# строим графики потерь и точности
N = np.arange(0, EPOCH)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


