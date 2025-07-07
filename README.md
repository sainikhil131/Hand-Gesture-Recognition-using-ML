
# 🤖 Hand Gesture Recognition using Machine Learning

This project demonstrates how to recognize hand gestures using Machine Learning and image processing techniques. It allows for collecting, training, and testing gesture datasets to classify different hand signs.


---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Main application file to run the gesture recognition system. |
| `collect-data.py` | Script to collect hand gesture images using a webcam. |
| `image_processing.py` | Contains functions for image preprocessing and enhancement. |
| `preprocessing.py` | Utility for additional preprocessing tasks before training. |
| `train.py` | Script to train the ML model using the collected gesture data. |
| `requirements_pip.txt` | Required Python packages for running the project using pip. |
| `requiremnets_conda.txt` | Required packages for setting up the environment using conda. |
| `signs.png` | Reference image showing the gesture labels used for training. |
| `README.md` | Documentation file (this file). |

---

## 🚀 Features

- Collect custom hand gesture datasets using webcam.
- Preprocess and clean images for training.
- Train a machine learning model for gesture classification.
- Real-time gesture recognition using webcam feed.

---

## 📦 Installation

1. Clone the repository:
 
   git clone https://github.com/your-username/Hand-Gesture-Recognition-using-ML.git
   cd Hand-Gesture-Recognition-using-ML


2. Create a virtual environment:

   Using pip:

 
   pip install -r requirements_pip.txt
  

   Or using conda:

  =
   conda create --name gesture_env --file requirements_conda.txt
   conda activate gesture_env
 

---

## 🧠 Model Training

1. Collect gesture images:

 
   python collect-data.py
   

2. Preprocess and clean images:

  
   python preprocessing.py
  

3. Train the model:

  
   python train.py
  

---

## 📷 Run the Application

Once the model is trained, you can start the live recognition:

```bash
python app.py
```

---

## 📸 Output

Below is a sample output from the live gesture recognition system:

![Sample Output](https://github.com/sainikhil131/Hand-Gesture-Recognition-using-ML/blob/cbb3e1aadfb9a177d9e92e69e5830b392ccb1cb5/1.jpg)

---

## ✍️ Author

* **GINJUPALLY SAI NIKHIL CHOWDARY**


---

