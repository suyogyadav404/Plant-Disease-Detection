import os
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st


# Define your model class
class SimpleCNN(nn.Module):
    def __init__(self, img_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (img_size // 4) * (img_size // 4), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (img_size // 4) * (img_size // 4))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set the image size and number of classes
img_size = 224
num_classes = 38 

# Load the trained model
model_path = r"Model/plant_disease_prediction_model.pth"
model = SimpleCNN(img_size=img_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# loading the class names
class_indices_path = r"Json_Files/class_indices.json"
class_indices = json.load(open(class_indices_path))


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(preprocessed_img)
        predicted_class_index = torch.argmax(predictions, axis=1).item()
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Home
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM ☘️")
    image_path = "App/image.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    **Welcome to the Plant Disease Detection System!** 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. 📤**Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. 🔎**Analysis:** By clicking on **Clasify** our system will process the image using advanced algorithms to identify potential diseases.

    ### Why Choose Us?
    - 📈**Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - 🤝**User-Friendly:** Simple and intuitive interface for seamless user experience.
    - ⚡**Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### 📢 About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About
elif(app_mode=="About"):
    st.header("📢 About")
    st.markdown("""
                #### About Dataset
                Download original dataset from Kaggle.
                
                (Url: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
                
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.
                
                -------------------------------------------------------------------------------------                
                ##### 👉**Team Members:** 
                ##### Suyog Yadav, Subhash Mote, Sonal Pohare, Vidya Patil, Mashuk Rabbani.

                """)
    
#Disease Detection
elif(app_mode=="Disease Detection"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection 🍂</h1>", unsafe_allow_html=True)
    st.markdown("<style> background-color: #f0f2f6;</style>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col3:
        st.image("https://img.freepik.com/premium-photo/indian-farmer-showing-money-cotton-field_75648-2061.jpg")

    with col1:
        st.image("https://t3.ftcdn.net/jpg/03/88/54/50/360_F_388545000_s4RsrD79y9GA04jkCsM8SX3wOaO9nSOW.jpg")

    with col2:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkIq9noK9xQWq6ID8_NJ-OCbllsIEp5UiatQ&s")
    

    uploaded_image = st.file_uploader("📤 **Upload an image...**", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
