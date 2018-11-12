import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D

### loading images local locations
lines = []
with open(r'.\driving_data\cc_run\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)
        
lines1 = []
with open(r'.\driving_data\cc_recoverrun\driving_log.csv') as csvfile:
    reader1 = csv.reader(csvfile)

    for line in reader1:
        lines1.append(line)
        
lines2 = []
with open(r'.\driving_data\cc_run1\driving_log.csv') as csvfile:
    reader2 = csv.reader(csvfile)

    for line in reader2:
        lines2.append(line)

images = []
measurements = []

for line in lines[0:6200]:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './driving_data/cc_run/IMG/' + filename
    image = cv2.imread(current_path)
#     print(type(image))
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print('image shape:', X_train.shape)
print('image shape:', y_train.shape)


### loading images data
images1 = []
measurements1 = []

for line in lines1:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './driving_data/cc_recoverrun/IMG/' + filename
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)

    images1.append(image)

    measurement = float(line[3])
    measurements1.append(measurement)


X_train1 = np.array(images1)
y_train1 = np.array(measurements1)

print('image shape:', X_train1.shape)
print('image shape:', y_train1.shape)

### loading images data
images2 = []
measurements2 = []

for line in lines2:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './driving_data/cc_run1/IMG/' + filename
    image = cv2.imread(current_path)
    images2.append(image)

    measurement = float(line[3])
    measurements2.append(measurement)


X_train2 = np.array(images2)
y_train2 = np.array(measurements2)

print('image shape:', X_train2.shape)
print('image shape:', y_train2.shape)

### concatenate data
combin_X_train = np.concatenate((np.concatenate((X_train, X_train1), axis=0),X_train2),axis=0)
combin_y_train = np.concatenate((np.concatenate((y_train, y_train1), axis=0),y_train2),axis=0)

print('combine_X_trainshape:', combin_X_train.shape)
print('')
print('combine_y_trainshape:', combin_y_train.shape)


model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

# model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(combin_X_train, combin_y_train, validation_split=0.2, epochs=5, shuffle=True)

model.save('model.h5')

