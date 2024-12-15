# Prediction script
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_and_prepare_image(img_path, target_size=(150, 150)): # Load an image file and prepare it for prediction
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_array, class_indices): # Predict the category of an image using the loaded model and display category.
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    class_names = {v: k for k, v in class_indices.items()}  
    predicted_category = class_names[predicted_class_index]
    return predicted_class_index, predicted_category

# Loading the model
model_path = 'C:\\wildfire_data\\wildfire_model.h5'
model = load_model(model_path)

# Defining class indices based on the dataset
class_indices = {'fire': 0, 'non_fire': 1, 'smoke': 2}  

# Path to the image we want to test
# img_path = 'path-of-the-image-file-for-prediction'

img_array = load_and_prepare_image(img_path)

# Prediction
predicted_class_index, predicted_category = predict_image(model, img_array, class_indices)
print("Predicted class index:", predicted_class_index)
print("Predicted category:", predicted_category)

