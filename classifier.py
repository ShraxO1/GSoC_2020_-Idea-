from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, BatchNormalization

classifier = Sequential()

classifier.add(Convolution2D(input_shape=(256, 256, 3), filters=96, kernel_size=(11, 11), strides=4, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters=256, kernel_size=(5, 5), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters=384, kernel_size=(3, 3), activation='relu'))
classifier.add(Convolution2D(filters=384, kernel_size=(3, 3), activation='relu'))
classifier.add(Convolution2D(filters=256, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=2048, activation='relu'))
classifier.add(Dense(output_dim=2048, activation='relu'))
classifier.add(Dense(output_dim=5, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('C:\\Users\\Shray Tyagi\\Downloads\\Waste-Classification-master\\Waste-Classification-master\\dataset\\train_set',
                                                 target_size=(256, 256),
                                                 batch_size=16
                                                 ,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('C:\\Users\\Shray Tyagi\\Downloads\\Waste-Classification-master\\Waste-Classification-master\\dataset\\test_set',
                                            target_size=(256, 256),
                                            batch_size=16,
                                            class_mode='categorical')

history = classifier.fit_generator(training_set,
                                   steps_per_epoch=3075,
                                   epochs=30,
                                   validation_data=test_set,
                                   validation_steps=228)

import numpy as np
from keras.preprocessing import image

pred = []
import matplotlib.pyplot as plt

for i in range(0, 15):
    img = image.load_img('C:\\Users\\Shray Tyagi\\Downloads\\Waste-Classification-master\\Waste-Classification-master\\check\\check (' + str(i) + ').jpg',
                         target_size=(256, 256))
    # plt.imshow(img)
    img = image.img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict_proba(img)
    pred.append(prediction)

training_set.class_indices

from keras.models import model_from_json

model_json = classifier.to_json()
with open("C:\\Users\\Shray Tyagi\\Downloads\\Waste-Classification-master\\Waste-Classification-master\\model_94%(256x256).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("C:\\Users\\Shray Tyagi\\.h5")
print("Saved model to disk")