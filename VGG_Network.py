import numpy as np
import sklearn.metrics
import keras.datasets
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential

#====================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#==========Functions============
import numpy as np

def Type_recognition(Fname):

    csvFileName1 = Fname.split('_')
    if (len(csvFileName1) == 9):

        if csvFileName1[6] == 'BL' or csvFileName1[6] == 'BR':  # bus
            return 0
        if csvFileName1[6] == 'CL' or csvFileName1[6] == 'CR':  # car
            return 1
        if csvFileName1[6] == 'ML' or csvFileName1[6] == 'MR':  # motorcycle
            return 2
        if csvFileName1[6] == 'TL' or csvFileName1[6] == 'TR':  # truck
            return 3
    else:
        csvFileName2 = csvFileName1[5].split('-')
        if csvFileName2[1] == 'BG':  # Noise
            return 4
        else:
            print('Invalid')

def Load_Data():
    file_name = "idmt_traffic_all.txt"
    CsvPath = 'C:/Users/Ladan_Gh/PycharmProjects/AI_Project/Feature/'

    #### Loading FileNames
    file = open(file_name, "r")
    list = file.readlines()
    y = np.zeros(len(list))
    x = np.zeros((len(list), 179, 87))
    c=0
    for i in list:
        Fname = i.split('.')
        F = Type_recognition(Fname[0])
        y[c] = F

        csvFileName4 = CsvPath + Fname[0] + '.csv'
        w_numpy = np.loadtxt(csvFileName4, delimiter=",", dtype=np.float64)
        x[c,:,:] = w_numpy[:, 0:]
        c=c+1

        #if c==2:
         #   break

    return x,y

Load_Data()
#==============================
# Model / data parameters
num_classes = 4
input_shape = (179, 87, 1)

#**************************
z = Load_Data()
x = z[0]
y = z[1]

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)

# ===================================
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# ========================================
model = Sequential()
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(179, 87, 1)))  # add my input_shape
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1))  # default stride is 2
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1))  # default stride is 2
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# ===================================
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# ===================================
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

#=====================================
score = model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test)
y_pred2=np.argmax(y_pred, axis=1)
y_test2=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test2, y_pred2)
print(cm)