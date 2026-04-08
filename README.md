# 🛡️ FraudShield AI — Enterprise Fraud Detection Platform

🚀 **Live Demo:** https://fraudshield-ai-05.streamlit.app/

💡 Designed as a **production-ready fintech fraud detection system**, not just a machine learning model.

> An end-to-end AI-powered fraud detection platform with real-time prediction, explainable AI, and an interactive dashboard.

---

## 🎯 Why This Project Stands Out

* 🔥 Real-time fraud detection system
* 🧠 Explainable AI (Why is this fraud?)
* ⚡ Live deployed dashboard
* 🏗️ Complete ML pipeline (not just model)
* 📊 Business-focused decision system

---

## 📋 Table of Contents

* [Business Use Case](#business-use-case)
* [Live Demo](#live-demo)
* [Architecture](#architecture)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Quick Start](#quick-start)
* [System Components](#system-components)
* [Model Performance](#model-performance)
* [API Reference](#api-reference)
* [Dashboard Guide](#dashboard-guide)
* [Screenshots](#screenshots)

---

## 🌐 Live Demo

👉 **Try the app here:**
🔗 https://fraudshield-ai-05.streamlit.app/

---

## 💼 Business Use Case

Credit card fraud costs the global financial industry **$32+ billion annually**.

Traditional systems fail due to:

* ❌ High false positives
* ❌ Poor interpretability
* ❌ Static rule-based systems

### ✅ FraudShield AI solves this:

| Challenge               | Solution                        |
| ----------------------- | ------------------------------- |
| Extreme class imbalance | SMOTE + undersampling + weights |
| Black-box models        | Explainable AI (local + global) |
| Batch processing        | Real-time fraud detection       |
| Static thresholds       | Dynamic threshold tuning        |

---

## 🏗️ Architecture

```
Data Ingestion → Feature Engineering → Model Training → Explainability
        ↓
Real-Time Engine → API Layer → Dashboard UI
```

---

## 🛠️ Tech Stack

| Layer          | Tools               |
| -------------- | ------------------- |
| Data           | Pandas, NumPy       |
| ML             | Scikit-learn        |
| Visualization  | Matplotlib, Seaborn |
| Backend        | FastAPI             |
| Frontend       | Streamlit           |
| Explainability | Custom XAI          |
| Storage        | joblib              |

---

## 📁 Project Structure

```
fraudshield-ai/
│── app/
│── src/
│── models/
│── logs/
│── images/
│── README.md
│── requirements.txt
│── train_pipeline.py
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

python train_pipeline.py

streamlit run app/dashboard.py

uvicorn app.api:app --port 8000
```

---

## 🔬 System Components

### 🔹 Data Ingestion

* Chunk-based loading (large datasets)
* Memory optimization
* Schema validation

### 🔹 Feature Engineering

* Normalization (Amount, Time)
* Velocity & rolling features
* Fraud pattern indicators

### 🔹 Modeling

* Logistic Regression
* Random Forest ⭐ (Best)
* Gradient Boosting

### 🔹 Explainable AI

* Feature importance
* Transaction-level reasoning
* Fraud scoring (0–100)

### 🔹 Real-Time Engine

* Live transaction simulation
* Risk categorization
* Alert system

---

## 📊 Model Performance

| Model               | Precision | Recall   | F1     |
| ------------------- | --------- | -------- | ------ |
| Logistic Regression | Low       | High     | Low    |
| Random Forest ⭐     | High      | Balanced | Best   |
| Gradient Boosting   | Medium    | High     | Medium |

---

## 🌐 API Reference

| Endpoint     | Method | Purpose            |
| ------------ | ------ | ------------------ |
| `/predict`   | POST   | Predict fraud      |
| `/explain`   | POST   | Explain prediction |
| `/metrics`   | GET    | Model metrics      |
| `/threshold` | POST   | Adjust threshold   |

---

## 🖥️ Dashboard Guide

* 📊 Fraud distribution
* 📈 ROC & PR curves
* 🔍 Feature importance
* ⚡ Real-time simulation
* 🚨 Fraud alerts

---

## 📸 Screenshots

### 📊 Dashboard

![Dashboard](images/dashboard.png)

### 📈 ROC Curve

![ROC](images/roc.png)

### 🔍 Feature Importance

![Feature](images/feature_importance.png)

### 📊 Model Comparison

![Model](images/model_comparison.png)

---

## 💡 Key Highlights

* 🧠 End-to-end ML system
* ⚡ Real-time fraud detection
* 🔍 Explainable AI integration
* 📊 Interactive dashboard
* 🚀 Live deployed product

---

## 👨‍💻 Author

**Tejas Pathak**
B.Tech CSE | AWS Certified
Aspiring Technical Project Manager

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
