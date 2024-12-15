import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Path for training and testing data
base_dir = 'C:/wildfire_data'  
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Data augmentation and setting the generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Model architecture with preprocessing
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=100,  
    epochs=7,  # should be optimally between 5 and 10
    validation_data=validation_generator,
    validation_steps=50
)

# Plots for training and validation
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('training & validation accuracy')
plt.legend()
plt.show()

# Test accuracy
test_loss, test_acc = model.evaluate(test_generator, steps=50)  # Adjust based on your test set size
print(f"test accuracy: {test_acc}")

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_acc}")
model.save('C:/wildfire_data/wildfire_model.h5')

# Class_indices using JSON function
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as json_file:
    json.dump(class_indices, json_file)

print("Model and class indices have been saved!")

# Loading the model
loaded_model = load_model('wildfire_model.h5')

# Image prediction script
def predict_image_category(model, img_path):
    # Loading and preprocessing the image for prediction
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    
    classes = train_generator.class_indices
    classes = dict((v, k) for k, v in classes.items())
    return classes[predicted_class_index[0]]

# Image path definition
# img_path = 'path-of-the-image-in-the-directory'

# Using this function to predict the category of the image
predicted_category = predict_image_category(loaded_model, img_path)
print("Predicted category:", predicted_category)


