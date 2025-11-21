# 🩺 Breast Cancer Prediction – Machine Learning Project

This repository contains a complete end-to-end machine learning project for predicting **Breast Cancer (Benign vs Malignant)** using Python, scikit-learn, and the Breast Cancer Wisconsin Diagnostic dataset.  
It includes data preprocessing, EDA, multiple ML models, evaluation metrics, and final results.

---

## 📁 Project Structure
breast-cancer-prediction/
│── breast-cancer-prediction.ipynb
│── README.md
│── requirements.txt
│── data/ (optional – dataset or link)
│── images/ (optional – plots, confusion matrix, ROC curve)
└── LICENSE (optional)

---

## 📊 Dataset
**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset  
**Source:** sklearn / Kaggle  

It contains:
- 30 numerical features like radius, texture, area, compactness  
- Target values:  
  - 0 → Benign  
  - 1 → Malignant

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- Jupyter Notebook  

---

## 🔍 Workflow (Step-by-Step)

### **1️⃣ Data Loading**
- Load dataset from sklearn or CSV
- Inspect columns, shapes, target classes

### **2️⃣ EDA (Exploratory Data Analysis)**
- Feature distribution plots  
- Correlation heatmap  
- Boxplots  
- Pairplots  

### **3️⃣ Preprocessing**
- Handle missing values (if any)  
- Encode labels  
- Scale features using StandardScaler  
- Train-test split (80/20 or 70/30)

### **4️⃣ Model Training**
Models used:
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  

### **5️⃣ Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC-AUC  

### **6️⃣ Final Result**
👉 Best model achieved: **XX% accuracy**  
(Replace XX% with your real result)

---

## ▶️ Run This Project Locally

### **Clone the Repository**
```bash
git clone https://github.com/rai8053/breast-cancer-prediction
cd breast-cancer-prediction
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Breast-Cancer-Prediction.ipynb
