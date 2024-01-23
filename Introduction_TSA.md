# Introduction to Time Series Analysis

Contents
--
- [Taxonomy of Time Series Analysis Domain](#)
- [Best Practices for Forecasting Model Selection]
- [Simple and Classical Forecasting Methods]
- [Time Series to Supervised Learning Problem]
- [Deep Learning for Time Series Forecasting]

## Taxonomy of Time Series Analysis Domain

When we are presented with a new time series forecasting problem, there are many things to consider. The choice that we made directly impacts each step of the project from the design of a test harness to evaluate forecast models to the fundamental difficulty of the forecast problem that you are working on. It is possible to very quickly narrow down the options by working through a series of questions about our time series forecasting problem. By considering a few themes and questions within each theme, we narrow down the type of problem, test harness, and even choice of algorithms for our project.

### Framework Overview

Time series forecasting involves developing and using a predictive model on data where there is an ordered relationship between observations. Before we get started on our project, we can answer a few questions and greatly improve our understanding of the structure of your forecast problem, the structure of the model requires, and how to evaluate it.

The framework can be divided into eight parts:

1. Inputs vs. Outputs.
2. Endogenous vs. Exogenous.
3. Unstructured vs. Structured.
4. Regression vs. Classification.
5. Univariate vs. Multivariate.
6. Single-step vs. Multi-step.
7. Static vs. Dynamic.
8. Contiguous vs. Discontiguous.

### 1. Inputs vs. Outputs

Generally, a prediction problem involves using past observations to predict or forecast one or more possible future observations. The goal is to guess about what might happen in the future.

- **Inputs**: Historical data provided to the model in order to make a single forecast.
- **Outputs**: Prediction or forecast for a future time step beyond the data provided as input.

**What are the inputs and outputs for a forecast?**

### 2. Endogenous vs. Exogenous

The input data can be further subdivided in order to better understand its relationship to the output variable.

- **Endogenous**: Input variables that are influenced by other variables in the system and on which the output variable depends.
- **Exogenous**: Input variables that are not influenced by other variables in the system and on which the output variable depends.

Typically, a time series forecasting problem has endogenous variables (e.g. the output is a function of some number of prior time steps) and may or may not have exogenous variables. Often, exogenous variables are ignored given the strong focus on the time series. 

**What are the endogenous and exogenous variables?**

### 3. Regression vs. Classification

Depending on the outcome type we can decide which kind of predictive modeling is required:

- **Regression**: Forecast a numerical quantity.
- **Classification**: Classify as one of two or more labels.

**Are you working on a regression or classification predictive modeling problem?**

### 4. Unstructured vs. Structured

It is useful to plot each variable in a time series and inspect the plot looking for possible patterns. A time series for a single variable may not have any obvious pattern. We can think of a series with no pattern as unstructured, as in there is no discernible time-dependent structure. Alternately, a time series may have obvious patterns, such as a trend or seasonal cycles as structured. 

- **Unstructured**: No obvious systematic time-dependent pattern in a time series variable.
- **Structured**: Systematic time-dependent patterns in a time series variable (e.g. trend and/or seasonality).

**Are the time series variables unstructured or structured?**

### 5. Univariate vs. Multivariate

Based on how many variables are being measured over time:

- **Univariate**: One variable measured over time.
- **Multivariate**: Multiple variables measured over time.

The number of variables may differ between the inputs and outputs, e.g. the data may not be symmetrical. For example, we may have multiple variables as input to the model and only be interested in predicting one of the variables as output. In this case, there is an assumption in the model that the multiple input variables aid and are required in predicting the single output variable.

- **Univariate and Multivariate Inputs**: One or multiple input variables measured over time.
- **Univariate and Multivariate Outputs**: One or multiple output variables to be predicted.

**Are you working on a univariate or multivariate time series problem?**

### 6. Single-step vs. Multi-step

- **One-step**: Forecast the next time step.
- **Multi-step**: Forecast more than one future time steps.

**Do you require a single-step or a multi-step forecast?**

### 7. Static vs. Dynamic

- **Static** -  A forecast model is fit once and used to make predictions.
- **Dynamic** -  A forecast model is fit on newly available data prior to each prediction.

**Do you require a static or a dynamically updated model?**

### 8. Contiguous vs. Discontiguous

- **Contiguous** - A time series where the observations are uniform over time may be described as contiguous. Many time series problems have contiguous observations, such as one observation each hour, day, month or year.
- **Discontiguous** - A time series where the observations are not uniform over time may be described as discontiguous. The lack of uniformity of the observations may be caused by missing or corrupt values. It may also be a feature of the problem where observations are only made available sporadically or at increasingly or decreasingly spaced time intervals.

**Are your observations contiguous or discontiguous?**

---














