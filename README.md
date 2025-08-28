# Sign Language Detection (NullClass Internship Task 5)

## 📌 Project Overview
This project is part of the NullClass Internship.  
The goal is to build a **Sign Language Detection system** that can recognize selected sign language gestures from both **images and real-time video input**.  
The application is designed to run only between **6 PM and 10 PM**, as per the task requirements.  

## 🚀 Features
- Detects and classifies sign language gestures.
- Works with both **image upload** and **real-time webcam feed**.
- Time restriction: only functions between **6:00 PM – 10:00 PM**.
- User-friendly **GUI** built with Python.

## 🛠️ Tech Stack
- **Python 3.10**
- **TensorFlow / Keras** (model training & prediction)
- **OpenCV** (real-time video capture & image processing)
- **Tkinter** (GUI development)
- **NumPy, Pandas, Matplotlib** (data handling and visualization)

## 📂 Project Structure
Task5_SignLanguageDetection/
│── app.py # GUI Application
│── sign_language_model.keras # Trained Model
│── sign_language_detection.ipynb # Training Notebook
│── requirements.txt # Dependencies
│── README.md # Project Documentation


## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SignLanguageDetection.git
   cd SignLanguageDetection

📊 Dataset
The model was trained on publicly available Sign Language datasets.
(Dataset source: Mention Kaggle/official dataset link here)

🎯 Output
Upload an image or start real-time video capture.
The GUI will display the predicted sign if run during the allowed time (6 PM – 10 PM).

👨‍💻 Internship Details
Internship Provider: NullClass
Task: Task 5 – Sign Language Detection
Requirement: GUI-based system with time restrictions

✅ Submission Checklist
 Model file (.keras)
 Training Notebook (.ipynb)
 GUI script (app.py)
 Requirements file (requirements.txt)
 README file (README.md)

✍️ Developed as part of the NullClass Internship Program.
