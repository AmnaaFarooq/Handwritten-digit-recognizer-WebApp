# ✍️ Handwritten Digit Recognizer Web App

This project is a real-time web application that recognizes handwritten digits (0-9). It uses the **MNIST dataset** and compares different **Naive Bayes** approaches to find the most effective model for image classification.

## 🚀 Features
* **Interactive Canvas:** Draw numbers directly in your browser.
* **Real-time Prediction:** Uses a pre-trained model to identify digits instantly.
* **Model Comparison:** Explores both Gaussian and Bernoulli Naive Bayes.

## 🧠 Behind the Scenes
I experimented with two algorithms:
1. **Gaussian Naive Bayes:** A great baseline but sensitive to pixel variance.
2. **Bernoulli Naive Bayes:** Optimized for binarized data, providing more stable results for handwritten strokes.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Library:** Scikit-learn
* **Frontend:** Streamlit
* **Image Processing:** Pillow (PIL) & NumPy

## 🏃 How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
