# Time-Series-Analysis-and-Forecasting-with-Python 📈📉📊⏰

🤘 Welcome to the most comprehensive, one-stop guide on **Time-Series Analysis, Forecasting, and Machine Learning** using Python 👨🏻‍💻. 

This repository is designed to take you from a complete beginner to an advanced practitioner. Whether you are dealing with classic forecasting (ARIMA, Prophet), framing time series as supervised machine learning (XGBoost, LightGBM), diving into state-of-the-art Deep Learning (LSTMs, CNNs, Transformers), or tackling advanced industrial use-cases like **Anomaly Detection**, **Clustering**, and **Classification**—this guide has everything you need in one place 🫱🏻‍🫲🏼.

🚀 The contents are structured logically: starting with foundational EDA and statistical analysis, moving through classical methodologies, and transitioning 💥 into cutting-edge machine learning and AutoML (FLAML) frameworks.

Cheers!! 🍻

## 💡 Quick Model Comparison Guide

Not sure which model to use for your forecasting task? Here is a quick organic comparison of the approaches covered in this repository to help you decide:

| Approach / Model | Complexity | Best Suited For | Pros | Cons |
| :--- | :---: | :--- | :--- | :--- |
| **Statistical (ARIMA/SARIMAX)** | Low | Small datasets, clear trends/seasonalities. | Highly interpretable, strong mathematical foundation. | Fails on complex non-linear patterns; requires strict stationarity. |
| **Tree-Based ML (XGBoost/LGBM)** | Medium | Tabularized time series with many external features. | Fast, highly accurate, handles non-linearities well. | Requires manual feature engineering (lags, rolling stats). |
| **Deep Learning (LSTMs/CNNs)** | High | Large datasets, complex sequential patterns. | Learns temporal dependencies automatically. | Computationally expensive; "black box" nature. |
| **Transformers** | Very High | Massive datasets, long-term forecasting. | State-of-the-art accuracy; captures long-range context via Self-Attention. | Extremely data-hungry; prone to overfitting on small data. |
| **FBProphet** | Low | Business time series (holidays, daily/weekly seasonality). | Works out-of-the-box, handles missing data gracefully. | Less flexible for highly irregular or high-frequency data. |
| **AutoML (FLAML)** | Low (Auto) | When you need a robust baseline without manual tuning. | Automatically finds the best algorithm and hyperparameters. | Computationally heavy during the search phase. |

## Contents 📄🗒

- **[Datasets Info](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Datasets_Info.md)📋**

- **[Introduction to Time Series Analysis(Theory)](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Introduction_TSA.md)🕰**
  
     - Taxonomy of Time Series Analysis Domain
     - Best Practices for Forecasting Model Selection
     - Simple and Classical Forecasting Methods
     - Time Series to Supervised Learning Problem
     - Deep Learning for Time Series Forecasting

- **[Time Series Data Visualization](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_Visualization_Basics.ipynb)📉**

    - Plotting of Pandas Df
    - Adding title
    - Adding Axis label
    - X limits by slice
    - X limit by argument
    - Color and Style
    - X ticks spacing
    - Date formatting
    - Major and Minor axis values
    - Gridlines

- **[Time Series EDA](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_EDA.ipynb)📊**
    
    - Introduction with time series data
    - Time resampling
    - Time downsampling/upsampling
    - Time Shifting
    - forward shift
    - backward shift
    - Rolling window mean
    - Expanding window mean/cumulative mean

- **[Time Series Data Analysis](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Data_Analysis.ipynb)💹**

    - Introduction to statsmodels
    - Hodrick Prescott filter - Trend/cyclical components
    - Time Series Stationarity
    - Augmented Dickey-Fuller Test
    - Granger Causality Tests
    - Time series decomposition
    - Additive/multiplicative models
    - Moving Average
    - Simple Exponentially weighted moving average(EWMA)
    - Double EWMA
    - Holt-Winters Method(Triple EWMA)

- **[Time Series Classification](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Classification.ipynb)🩺**
    - Introduction with heartbeat classification
    - Traditional ML classifier (Feature extraction + Random Forest)
    - Deep Learning classifier (1D CNN in PyTorch)
    - Signal visualization and evaluation metrics (Confusion Matrix, ROC Curve)

- **[Time Series Anomaly Detection](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Anomaly_Detection.ipynb)🚨**
    - Synthetic anomaly injection on real-world data
    - Unsupervised Anomaly Detection using Isolation Forest
    - Deep Learning Anomaly Detection using PyTorch Autoencoders (Reconstruction Loss)

- **[Time Series Clustering](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Clustering.ipynb)🗂️**
    - Dynamic Time Warping (DTW) distance metric
    - K-Means with DTW using `tslearn`
    - Grouping heartbeat sequences by shape profiles

- **[Time Series Feature Engineering & Machine Learning](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Feature_Engineering_and_ML.ipynb)🤖**
    - Re-framing time series as supervised learning
    - Advanced Feature Engineering (Lags, rolling stats, cyclical time encoding)
    - Time-series cross validation (TimeSeriesSplit)
    - Tree-based regressors (XGBoost, LightGBM)
    - Walk-forward validation evaluation

- **[Time Series Forecasting Classical Methods](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_Traditional_Methods.ipynb)🤖**

    - Forecasting with Holts-Winter Method
    - Autocorrelation function(ACF)
    - Partial autocorrelation function(PACF)
    - Autocovariance for 1D
    - Autocorrelation for 1D
    - Autoregressive model(AR(p))
    - Autoregressive Moving Average(ARMA) Model
    - Autoregressive Integrated Moving Average(ARIMA)
    - Error/Trend/Seasonal Decomposition(ETS Decomposition)
    - Seasonal Autoregressive Integrated Moving Averages(SARIMA)
    - Seasonal AutoRegressive Integrated Moving Average with EXogenous Variable.

- **[Time Series Forecasting with Deep Learning](#)🕸️**

    - [MLPs for time series forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_MLPs.ipynb)
    - [LSTMs for time series forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_LSTMs.ipynb)
    - [CNNs for time series forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_CNNs.ipynb)
    - [Transformers for time series forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_Transformers.ipynb)
 
- **[Time Series Forecasting with FBProphet](#)🎯**
    -  [Univariate and Multivariate Time Series Forecasting With FBProphet](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Time_Series_Forecasting_With_Prophet.ipynb)

- **[AutoML For Time Series Forecasting](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Automating_Time_Series_Forecasting_with_FLAML.ipynb)🦾**
    -  [Automating Time Series Forecasting with FLAML](https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python/blob/master/Automating_Time_Series_Forecasting_with_FLAML.ipynb)