# 🛰️ Radar (SAR) Target Recognition System
### Deep Learning | Computer Vision | Military Intelligence

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)

## 📝 Project Overview
This project is a high-performance **Deep Learning application** designed to recognize military targets from **Synthetic Aperture Radar (SAR)** imagery. Using the **MSTAR Dataset**, I developed a Convolutional Neural Network (CNN) that can identify armored vehicles with high precision, even in complex environments.

## 🚀 Key Features
- **Accurate Classification:** Identifies 10 different military vehicle classes (T-72, BMP-2, BTR-70, etc.).
- **Interactive Web UI:** Users can upload SAR images and get instant predictions via a **Streamlit** dashboard.
- **Optimized for Linux:** Developed and tested on **Linux Mint** for stable performance.
- **Real-time Visualization:** Displays the target image alongside the AI's confidence score.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow & Keras (CNN Architecture)
- **Frontend:** Streamlit (Web Application)
- **Data Processing:** NumPy, OpenCV, Pillow
- **Development Environment:** Linux Mint (ThinkPad)

## 📂 Repository Structure
| File | Description |
| :--- | :--- |
| `app.py` | Main application script for the Web UI |
| `mstar_pro_model.h5` | Pre-trained CNN model weights |
| `requirements.txt` | Necessary Python libraries for the project |
| `README.md` | Project documentation (this file) |

## 📊 Dataset
The model uses the **MSTAR (Moving and Stationary Target Acquisition and Recognition)** dataset. 
> [!IMPORTANT]
> Due to file size constraints, the raw dataset is not hosted here. You can access it on [Kaggle](https://www.kaggle.com/datasets/a970932010/mstar-dataset).

## ⚙️ How to Setup
1. **Clone the Repo:** `git clone https://github.com/yazdan-yousaf/radar-sar-recognition.git`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run the App:** `streamlit run app.py`

---
**Developed with ❤️ by Jalal** *Passionate about AI, Linux, and Computer Vision.*
