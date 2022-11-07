#Import ImageDataGenerator Library And Configure It
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
# Apply ImageDataGenerator Functionality To Train And Test Set
x_train = train_datagen.flow_from_directory('/workspace/IBM-Project-31588-1660203186/Project Development Phase/Sprint 1/Data Collection/Create Train And Test Folders/Dataset/training_set', target_size=(64, 64), batch_size=300, class_mode='categorical', color_mode='grayscale')
x_test = test_datagen.flow_from_directory('/workspace/IBM-Project-31588-1660203186/Project Development Phase/Sprint 1/Data Collection/Create Train And Test Folders/Dataset/test_set', target_size=(64, 64), batch_size=300, class_mode='categorical', color_mode='grayscale')
# Import The Required Model Building Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
# Initialize The Model
model = Sequential();
# Add The Convolution Layer
model.add(Convolution2D(32, (3,3), input_shape=(64,64,1), activation='relu'))
# Add The Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Add The Flatten Layer
model.add(Flatten())
# Adding The Dense Layers
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=9, activation='relu'))
# Compile The Model
model.compile(loss='categorical_entropy', optimizer='adam', metrics=['accuracy'])
# Fit the Model
model.fit_generator(x_train, steps_per_epoch=24, epochs=10, validation_data=x_test, validation_steps=40)
# Save the model
model.save('aslpng1.h5')