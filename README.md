# Malaria Cell Classifier (AI in Medical Imaging)

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify red blood cell images as **Parasitized (infected)** or **Uninfected**, aiding in rapid malaria screening.

## The Problem
Malaria is a life-threatening disease. Manual microscopic diagnosis is time-consuming and requires highly trained experts. This project explores how a CNN can assist in automating this screening process.

## Results & Key Clinical Insight
- **Overall Accuracy:** ~95% on the validation set.
- **The False Negative Problem:** The model achieved a **Recall of 0.94** for infected cells. This means it misses roughly 6% of actual infections. In clinical screening, a False Negative (sending a sick patient home) is far more dangerous than a False Positive (asking a healthy patient to retake the test). Future iterations should lower the classification threshold (e.g., from 0.5 to 0.3) to prioritize **Recall** over **Precision**, ensuring fewer infected patients are missed.

## Tech Stack & Architecture
- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Architecture:** A 3-layer CNN with Max Pooling, followed by Dense layers.
- **Dataset:** NIH Malaria Cell Images (via Kaggle) - 27,560 images.

##<img width="1172" height="508" alt="Screenshot 2026-11-16 214831" src="https://github.com/user-attachments/assets/cdb8693a-1094-4934-be14-ecd6bf8bf7b7" />
<img width="702" height="822" alt="Screenshot 2026-11-16 214821" src="https://github.com/user-attachments/assets/583070f3-44d6-4835-9d04-232545cbc25b" />
## How to Run
1. Open the `.ipynb` file in Google Colab.
2. Run all cells. The dataset will download automatically via `kagglehub`.
