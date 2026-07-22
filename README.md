---
title: Keratoconus Multi-Map Classifier
emoji: 👁️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# AI-Powered Multi-Map Framework for Keratoconus Classification

Upload 7 corneal maps (SAG_A, SAG_P, ELV_A, ELV_P, CT_A, EC_A, EC_P) and get a prediction.

## Features

- Multi-map keratoconus classification
- MobileNet-V3-Large feature extraction
- StandardScaler + SelectKBest + SVM classification pipeline
- AI saliency heatmap visualization
- PDF report generation
- FastAPI web application

## Required Maps

1. SAG_A
2. SAG_P
3. ELV_A
4. ELV_P
5. CT_A
6. EC_A
7. EC_P

## Technology Stack

- FastAPI
- PyTorch
- Torchvision
- Scikit-learn
- OpenCV
- NumPy
- FPDF2

This application predicts keratoconus from seven corneal topography maps using a hybrid deep-learning and machine-learning pipeline.