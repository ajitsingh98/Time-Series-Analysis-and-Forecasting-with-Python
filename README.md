# Time-Series Analysis and Forecasting with Python 📈📉

Welcome to a comprehensive guide on **Time-Series Analysis, Forecasting, and Machine Learning** using Python. 

This repository is designed to take you from a foundational understanding to advanced practice. Whether you are dealing with classical forecasting (ARIMA, Prophet), framing time series as supervised machine learning (XGBoost, LightGBM), diving into state-of-the-art Deep Learning (LSTMs, CNNs, Transformers), or tackling advanced industrial use-cases like **Anomaly Detection**, **Clustering**, and **Classification**—this repository provides practical, code-first implementations.

The contents are structured logically: starting with foundational Exploratory Data Analysis (EDA) and statistical analysis, moving through classical methodologies, and transitioning into cutting-edge machine learning and AutoML frameworks.

## 💡 Quick Model Comparison Guide

Not sure which model to use for your forecasting task? Here is a high-level comparison of the approaches covered in this repository to help you decide:

| Approach / Model | Complexity | Best Suited For | Pros | Cons |
| :--- | :---: | :--- | :--- | :--- |
| **Statistical (ARIMA/SARIMAX)** | Low | Small datasets, clear trends/seasonalities. | Highly interpretable, strong mathematical foundation. | Fails on complex non-linear patterns; requires strict stationarity. |
| **Tree-Based ML (XGBoost/LGBM)** | Medium | Tabularized time series with external features. | Fast, highly accurate, handles non-linearities well. | Requires manual feature engineering (lags, rolling stats). |
| **Deep Learning (LSTMs/CNNs)** | High | Large datasets, complex sequential patterns. | Learns temporal dependencies automatically. | Computationally expensive; "black box" nature. |
| **Transformers** | Very High | Massive datasets, long-term forecasting. | State-of-the-art accuracy; captures long-range context. | Extremely data-hungry; prone to overfitting on small data. |
| **FBProphet** | Low | Business time series (holidays, daily/weekly seasonality). | Works out-of-the-box, handles missing data gracefully. | Less flexible for highly irregular or high-frequency data. |
| **AutoML (FLAML)** | Low | Robust baselines without manual tuning. | Automatically finds the best algorithm and hyperparameters. | Computationally heavy during the search phase. |

## 📑 Contents

### 1. 🏗️ Fundamentals and Analysis
Build a strong foundation in time-series concepts, visualization, and statistical analysis.
- 📋 **[Datasets Info](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Datasets_Info.md)**
- 📚 **[Introduction to Time Series Analysis (Theory)](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Introduction_TSA.md)**
- 📊 **[Time Series Data Visualization](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_Visualization_Basics.ipynb)**
- 🔬 **[Time Series EDA](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_EDA.ipynb)**
- 📉 **[Time Series Data Analysis](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_Analysis.ipynb)**

### 2. 🔮 Forecasting & Machine Learning
Forecast future values using both classical statistical techniques and modern machine learning.
- 📈 **[Time Series Forecasting Classical Methods](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_Traditional_Methods.ipynb)**
- ⚙️ **[Time Series Feature Engineering & Machine Learning](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Feature_Engineering_and_ML.ipynb)**
- 🎯 **[Time Series Forecasting with FBProphet](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_Prophet.ipynb)**
- 🦾 **[AutoML For Time Series Forecasting (FLAML)](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Automating_Time_Series_Forecasting_with_FLAML.ipynb)**

### 3. 🧠 Deep Learning
Leverage neural networks to capture complex, non-linear relationships in sequential data.
- 🕸️ **[MLPs for Time Series Forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_MLPs.ipynb)**
- 🔁 **[LSTMs for Time Series Forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_LSTMs.ipynb)**
- 🖼️ **[CNNs for Time Series Forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_CNNs.ipynb)**
- 🤖 **[Transformers for Time Series Forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_Transformers.ipynb)**

### 4. 🚀 Advanced Time Series Tasks
Go beyond forecasting to tackle real-world industrial problems.
- 🗂️ **[Time Series Classification](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Classification.ipynb)**
- 🚨 **[Time Series Anomaly Detection](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Anomaly_Detection.ipynb)**
- 🧩 **[Time Series Clustering](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Clustering.ipynb)**

---
*Created and maintained by [Ajit Kumar Singh](https://github.com/ajitsingh98).*