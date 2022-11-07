# Image Preprocessing

# Import ImageDataGenerator Library And Configure It
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Apply ImageDataGenerator Functionality To Train And Test Set
x_train = train_datagen.flow_from_directory('/workspace/IBM-Project-31588-1660203186/Project Development Phase/Sprint 1/Data Collection/Create Train And Test Folders/Dataset/training_set', target_size=(64,64),batch_size=300, class_mode='categorical', color_mode ="grayscale")
x_test = test_datagen.flow_from_directory('/workspace/IBM-Project-31588-1660203186/Project Development Phase/Sprint 1/Data Collection/Create Train And Test Folders/Dataset/test_set', target_size=(64,64),batch_size=300, class_mode='categorical', color_mode ="grayscale")