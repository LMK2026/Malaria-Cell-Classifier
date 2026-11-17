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
<img width="730" height="838" alt="Screenshot 2026-11-16 223515" src="https://github.com/user-attachments/assets/97f06f8d-2fcc-4966-ac24-6aed76dbb943" />
<img width="1560" height="692" alt="Screenshot 2026-11-16 223456" src="https://github.com/user-attachments/assets/7268d746-b4d4-4880-884e-2cc2e1d348e1" />

## How to Run
1. Open the `.ipynb` file in Google Colab.
2. Run all cells. The dataset will download automatically via `kagglehub`.
