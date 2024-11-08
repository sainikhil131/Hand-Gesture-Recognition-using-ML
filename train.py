# Importing the necessary libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the environment variable to specify the GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define image size and batch size
sz = 128
batch_size = 10
eps = 20  # Define the number of epochs

# Step 1 - Building the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers with dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))

# Load training and testing sets, checking the number of samples
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(sz, sz),
                                                 batch_size=batch_size,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test/train',
                                            target_size=(sz, sz),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Print the class indices to confirm the number of classes
print(training_set.class_indices)
print(test_set.class_indices)

# Modify the output layer to match the dataset's class count
num_classes = len(training_set.class_indices)
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Print number of samples to verify data loading
print("Training samples:", training_set.n)
print("Testing samples:", test_set.n)

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = training_set.n // training_set.batch_size
validation_steps = test_set.n // test_set.batch_size

# Check if data is loaded
if training_set.n > 0 and test_set.n > 0:
    # Training the model
    classifier.fit(
        training_set,
        validation_data=test_set,
        epochs=eps,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    # Saving the model
    model_json = classifier.to_json()
    with open("model/model-bw.json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights('model/model-bw.weights.h5')
    print('Model and weights saved')
else:
    print("Error: No images found in the specified directories. Please check the folder paths and structure.")
