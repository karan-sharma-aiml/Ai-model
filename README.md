# ♻️ AI Waste Classification using MobileNetV2

A deep learning based waste classification system built with **TensorFlow/Keras** using **MobileNetV2 Transfer Learning**. The model classifies waste into six categories to support smart waste segregation and recycling systems.

---

## 🚀 Features

- MobileNetV2 Transfer Learning
- Leakage-safe Train / Validation / Test Split
- Duplicate Image Detection & Removal
- Data Augmentation Pipeline
- Temperature Scaling Calibration
- Balanced Class Weights
- Production-ready Prediction Pipeline
- Comprehensive Evaluation Metrics
- Clean & Modular Code Structure

---

## 📂 Dataset

**Dataset:** TrashNet

| Property | Value |
|----------|------:|
| Images | 2527 |
| Classes | 6 |
| Image Size | 224 × 224 |

### Classes

- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

## 🧠 Model

- Backbone : MobileNetV2
- Framework : TensorFlow 2.21
- Optimizer : AdamW
- Loss : Categorical Crossentropy
- Fine-Tuning : Final Backbone Layers
- Calibration : Temperature Scaling

---

## 📊 Performance

| Metric | Score |
|---------|-------:|
| Accuracy | **80.74%** |
| Precision (Weighted) | **81.06%** |
| Recall (Weighted) | **80.74%** |
| F1 Score (Weighted) | **80.66%** |
| Top-2 Accuracy | **92.88%** |
| Top-3 Accuracy | **97.63%** |

---

## 📁 Project Structure

```text
AI_MODEL/
│
├── saved_model/
│   └── best_model.keras
│
├── train.py
├── predict.py
├── evaluate.py
├── waste_pipeline.py
├── requirements.txt
└── README.md
```

---

## ▶️ Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Predict Image

```bash
python predict.py
```

### Evaluate Model

```bash
python evaluate.py
```

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Scikit-learn
- Matplotlib
- Pillow

---

## 📌 Future Improvements

- Vision Transformer (ViT)
- EfficientNetV2
- Real-time Webcam Prediction
- Web Deployment (FastAPI + React)
- Mobile Application
- Multi-object Waste Detection

---

## 👨‍💻 Author

**Karan Sharma**

B.Tech – Artificial Intelligence & Machine Learning

GitHub: https://github.com/karan-sharma-aiml
