# Deep Learning Applications in Finance & Predictive Maintenance

**Developer**: Gharib Ayman
**Supervisor**: Prof. EL ACHAAK Lotfi  

![Applications](https://img.shields.io/badge/Applications-Finance_&_Predictive_Maintenance-blue) 
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)

## 📌 Project Overview
This project showcases deep learning's effectiveness in two key areas:
- **Stock Price Forecasting** using time-series regression.
- **Predictive Maintenance** for industrial equipment failure classification.

The implementation covers complete workflows from data preprocessing to model evaluation, emphasizing high-accuracy predictive analytics in finance and industrial systems.

---

## 📈 Stock Price Forecasting with Deep Learning

### 🎯 Objective
Predict stock closing prices based on historical market data to enhance investment decisions.

### 📊 Dataset
- **Source**: [NYSE Kaggle Dataset (2010–2017)](https://www.kaggle.com/datasets/nyse-data)
- **Features**: Open/Close prices, High/Low values, Volume, Stock Symbols
- **Target**: Closing price (`close`)

### 🛠 Implementation Details
#### **Preprocessing**
- Temporal feature engineering (Moving Averages, RSI, MACD)
- Time-aware data split (70-15-15)
- Min-Max normalization

#### **Model Architecture**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])
```
- **Optimizer**: Adam (`lr=0.001`, decay)
- **Loss**: MAE + Early Stopping (`patience=10`)

#### **📊 Results**
| Metric | Value |
|--------|-------|
| RMSE | $2.87 |
| R² | 0.94 |
| MAE | $1.62 |

#### **Key Insights**
- LSTM outperformed MLP by **18%** in RMSE.
- MACD indicators improved volatility handling.

---

## ⚙️ Predictive Maintenance Classification

### 🎯 Objective
Classify equipment failure risks using sensor data to optimize maintenance schedules.

### 📊 Dataset
- **Source**: [Machine Sensor Data on Kaggle](https://www.kaggle.com/)
- **Features**: Temperature, Torque, Tool Wear (10 parameters)
- **Classes**: 6 failure modes + normal operation

### 🛠 Implementation Details
#### **Preprocessing**
- SMOTE for class imbalance handling
- Random Forest-based feature selection
- Stratified 5-fold data split

#### **Model Architecture**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(n_timesteps, n_features)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```
- **Loss**: Focal Loss (`γ=2, α=0.25`)
- **Optimizer**: Nadam

#### **📊 Results**
| Metric | Value |
|--------|-------|
| Accuracy | 92.4% |
| Macro F1 | 0.89 |
| AUC-ROC | 0.97 |

#### **Key Findings**
- Tool wear & temperature variance were top predictors.
- Reduced false negatives by **22%** using Focal Loss.

---

## 🔍 Comparative Analysis
| Aspect | Stock Prediction | Predictive Maintenance |
|--------|-----------------|----------------------|
| **Data Type** | Time-Series | Multivariate Tabular |
| **Primary Challenge** | Non-stationarity | Class Imbalance |
| **Optimal Model** | Seq2Seq LSTM | 1D CNN + Focal Loss |
| **Critical Metric** | MAE | Recall |

---

## 🚀 Conclusion & Next Steps
### **Impact**
- **Financial Forecasting**: Enables **proactive portfolio management**, achieving **94% R²** accuracy in price predictions.
- **Industrial IoT**: Reduces maintenance costs by **~35%**, with a **97% AUC-ROC** for failure detection.

### **Future Enhancements**
✅ **Deploy as a Flask API for real-time inference**
✅ **Implement concept drift adaptation for evolving market conditions**
✅ **Optimize for edge deployment on IoT devices**

---

💡 _If you find this project useful, feel free to ⭐ the repository!_
