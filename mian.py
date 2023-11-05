import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the directory containing the training and validation data
train_data_dir = 'path/to/training/data'
validation_data_dir = 'path/to/validation/data'

# Set the parameters for the CNN
img_width, img_height = 48, 48
batch_size = 32
epochs = 20
num_classes = 4  # Number of emotions to classify (happiness, sadness, anger, surprise)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Data augmentation for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

# Load and prepare the validation data
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                              batch_size=batch_size, class_mode='categorical')

# Train the model
model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size,
                    epochs=epochs, validation_data=validation_generator,
                    validation_steps=validation_generator.n // batch_size)

# Save the trained model
model.save('emotion_recognition_model.h5')
