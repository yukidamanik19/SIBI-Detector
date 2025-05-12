# 🤟 SIBI Detector – Indonesian Sign Language Recognition System

A final-year capstone project focused on building a real-time Indonesian Sign Language (SIBI) recognition system using **MediaPipe** and **custom-trained gesture recognition models**. This project aims to improve accessibility for the deaf community by translating hand gestures into readable text through a web-based AI application.

---

## 🔍 Overview

**SIBI Detector** is a web application that captures and classifies hand gestures from a webcam feed using MediaPipe and displays the translated output in Bahasa Indonesia. It supports live detection of custom-trained gestures for common SIBI words.

---

## 🚀 Features

- Real-time hand gesture detection with MediaPipe
- Custom model trained for specific SIBI gestures
- Translation of gestures into text
- Web interface with camera input
- Responsive and easy-to-use interface
- Accessible and inclusive design

---

## 🛠️ Tech Stack

- **Python**
- **MediaPipe** (Gesture Recognition, Hand Landmark)
- **TensorFlow Lite**
- **Flask** (Web Framework)
- **HTML/CSS/JavaScript**
- **Node.js (Optional for frontend extensions)**

---Berikut adalah isi lengkap `README.md` yang sudah disesuaikan dengan proyek **SIBI Detector**, dirancang agar terlihat profesional di GitHub:

---

````markdown
# 🤟 SIBI Detector – Indonesian Sign Language Recognition System

A final-year capstone project focused on building a real-time Indonesian Sign Language (SIBI) recognition system using **MediaPipe** and **custom-trained gesture recognition models**. This project aims to improve accessibility for the deaf community by translating hand gestures into readable text through a web-based AI application.

---

## 🔍 Overview

**SIBI Detector** is a web application that captures and classifies hand gestures from a webcam feed using MediaPipe and displays the translated output in Bahasa Indonesia. It supports live detection of custom-trained gestures for common SIBI words.

---

## 🚀 Features

- Real-time hand gesture detection with MediaPipe
- Custom model trained for specific SIBI gestures
- Translation of gestures into text
- Web interface with camera input
- Responsive and easy-to-use interface
- Accessible and inclusive design

---

## 🛠️ Tech Stack

- **Python**
- **MediaPipe** (Gesture Recognition, Hand Landmark)
- **TensorFlow Lite**
- **Flask** (Web Framework)
- **HTML/CSS/JavaScript**
- **Node-RED** *(for integration purposes, if used)*

---

## 📁 Project Structure

```bash
SIBI-Detector/
│
├── app.py                     # Main Flask backend
├── gesture_recognizer.task    # Trained TFLite model
├── static/                    # Static assets (JS/CSS)
├── templates/                 # HTML templates
├── script.js                  # Frontend camera & logic
├── Training_model_mediapipe.ipynb  # Model training notebook
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
````

---

## 🧠 How It Works

1. The system accesses the camera feed via a web browser.
2. MediaPipe detects hand landmarks and classifies them using a trained gesture model.
3. Recognized gestures are mapped to predefined SIBI vocabulary.
4. The translation is displayed in real-time on the web interface.

---

## ✅ Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
```

Then open your browser at: `http://localhost:5000`

---

## 📸 Screenshots

*Add screenshots or GIFs of your running app interface here to demonstrate functionality.*

---

## 🌐 Acknowledgments

* [Google's MediaPipe](https://mediapipe.dev/)
* [TensorFlow Lite](https://www.tensorflow.org/lite)
* Universitas Prasetiya Mulya – Capstone Project 2025

```

---

Kamu bisa langsung **copy-paste** isi ini ke file `README.md` di folder proyek kamu sebelum meng-upload ke GitHub.

Kalau kamu mau, saya juga bisa bantu buatkan file `.md`-nya dan mengirimkannya langsung. Mau?
```

