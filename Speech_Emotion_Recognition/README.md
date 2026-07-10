# 🎤 Speech Emotion Recognition using Deep Learning

## 📌 Project Overview

Speech Emotion Recognition (SER) is a deep learning application that identifies human emotions from speech recordings. This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset and Mel-Frequency Cepstral Coefficients (MFCCs) to classify speech into eight different emotional categories.

A Deep Neural Network (DNN) was trained on extracted MFCC features to recognize emotions from speech audio. The project also includes a Streamlit web application for real-time emotion prediction.

---

## 🚀 Features

- Speech Emotion Recognition using Deep Learning
- Audio preprocessing using Librosa
- MFCC feature extraction
- Exploratory Data Analysis (EDA)
- Deep Neural Network built with TensorFlow/Keras
- Model evaluation using Accuracy, Classification Report, and Confusion Matrix
- Saved trained model for future inference
- Streamlit web application for predictions

---

## 📂 Dataset

Dataset Used:

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

The dataset contains speech recordings from 24 professional actors expressing different emotions.

Recognized emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📁 Project Structure

```text
Speech_Emotion_Recognition/
│
├── dataset/
├── images/
├── model/
│   ├── speech_emotion_dnn.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── Speech_Emotion_Recognition.ipynb
├── app.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/codealpha_tasks.git
```

Move into the project folder:

```bash
cd codealpha_tasks/Speech_Emotion_Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Notebook

Open:

```
Speech_Emotion_Recognition.ipynb
```

Run all cells sequentially to:

- Load the dataset
- Perform EDA
- Extract MFCC features
- Train the Deep Neural Network
- Evaluate the model
- Save the trained model

---

## 🌐 Running the Streamlit Application

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

The Deep Neural Network consists of:

- Dense Layer (128 neurons, ReLU)
- Dropout (0.3)
- Dense Layer (64 neurons, ReLU)
- Dropout (0.3)
- Dense Layer (32 neurons, ReLU)
- Output Layer (8 neurons, Softmax)

---

## 📊 Model Performance

Training Accuracy:

**~77%**

Validation Accuracy:

**~61%**

Test Accuracy:

**~58%**

The model successfully learns meaningful emotional patterns from speech recordings. While performance can be further improved using CNNs, LSTMs, or larger feature representations, this project demonstrates the complete workflow of a Speech Emotion Recognition system.

---

## 🔮 Future Improvements

Possible improvements include:

- CNN-based Speech Emotion Recognition
- LSTM / GRU models
- Data augmentation
- Hyperparameter tuning
- Early stopping and model checkpointing
- Real-time microphone input
- Probability visualization in the Streamlit application

---

## 👨‍💻 Author

Developed as part of the **CodeAlpha Machine Learning Internship** using Python, TensorFlow, and Streamlit.