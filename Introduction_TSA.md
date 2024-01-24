# Introduction to Time Series Analysis

Contents
--
- [Taxonomy of Time Series Analysis Domain](#taxonomy-of-time-series-analysis-domain)
- [Best Practices for Forecasting Model Development](#best-practices-for-forecasting-model-development)
- [Simple and Classical Forecasting Methods](#simple-and-classical-forecasting-methods)
- [Time Series to Supervised Learning Problem](#time-series-to-supervised-learning-problem)
- [Deep Learning for Time Series Forecasting](#deep-learning-for-time-series-forecasting)

## Taxonomy of Time Series Analysis Domain

Topics:

- A structured way of thinking about time series forecasting problems.
- A framework to uncover the characteristics of a given time series forecasting problem.
- A suite of specific questions, the answers to which will help to define your forecasting problem.

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
- **Discontiguous** - A time series where the observations are not uniform over time may be described as discontiguous. The lack of uniformity of the observations may be ca used by missing or corrupt values. It may also be a feature of the problem where observations are only made available sporadically or at increasingly or decreasingly spaced time intervals.

**Are your observations contiguous or discontiguous?**

---


## Best Practices for Forecasting Model Development

Topics:

- A systematic four-step process that you can use to work through any time series forecasting problem.
- A list of models to evaluate and the order in which to evaluate them.
- A methodology that allows the choice of a final model to be defensible with empirical evidence, rather than whim or fashion.

The mistake that almost all beginners make is going straight to modeling without a strong idea of what problem is being solved or how to robustly evaluate candidate solutions. This almost always results in a lot of wasted time. Slow down, follow the process, and complete each step.

The typical process is divided into four parts; they are:

1. Define Problem.
2. Design Test Harness. 
3. Test Models.
4. Finalize Model.


#### Step 1: Define Problem

Define your time series problem. Some topics to consider and motivating questions within each topic are as follows:

1. *Inputs vs. Outputs*: What are the inputs and outputs for a forecast?
2. *Endogenous vs. Exogenous*: What are the endogenous and exogenous variables?
3. *Unstructured vs. Structured*: Are the time series variables unstructured or structured?
4. *Regression vs. Classification*: Are you working on a regression or classification predictive modeling problem? What are some alternate ways to frame your time series forecasting problem?
5. *Univariate vs. Multivariate*: Are you working on a univariate or multivariate time series problem?
6. *Single-step vs. Multi-step*: Do you require a single-step or a multi-step forecast?
7. *Static vs. Dynamic*: Do you require a static or a dynamically updated model?
8. *Contiguous vs. Discontiguous*: Are your observations contiguous or discontiguous?

Some useful tools to help get answers include:
- Data visualizations (e.g. line plots, etc.)
- Statistical analysis (e.g. ACF/PACF plots, etc.)
- Domain experts
- Project stakeholders

#### Step 2: Design Test Harness

Below is a common time series forecasting model evaluation scheme if you are looking for ideas:

1. Split the dataset into a train and test set.
2. Fit a candidate approach on the training dataset.
3. Make predictions on the test set directly or using walk-forward validation.
4. Calculate a metric that compares the predictions to the expected values.

The test harness must be robust and you must have complete trust in the results it provides. An important consideration is to ensure that any coefficients used for data preparation are estimated from the training dataset only and then applied on the test set. This might include mean and standard deviation in the case of data standardization.

#### Step 3: Test Models - Model Selection

Test many models using your test harness. Some common classes of methods that you can design experiments around include the following:


1. **Baseline** -  Simple forecasting methods such as persistence and averages.
2. **Autoregression** - The Box-Jenkins process and methods such as SARIMA.
3. **Exponential Smoothing** - Single, double and triple exponential smoothing methods.
4. **Linear Machine Learning** - Linear regression methods and variants such as regularization.
5. **Nonlinear Machine Learning** - kNN, decision trees, support vector regression and more.
6. **Ensemble Machine Learning** - Random forest, gradient boosting, stacking and more. 7. Deep Learning. MLPs, CNNs, LSTMs, and Hybrid models.

Order here is important and is structured in increasing complexity from classical to modern methods. Early approaches are simple and give good results fast; later approaches are slower and more complex, but also have a higher bar to clear to be skillful.

This list is based on a univariate time series forecasting problem, but you can adapt it for the specifics of your problem, e.g. use VAR/VARMA/etc. in the case of multivariate time series forecasting.

Some data preparation schemes to consider include:

- Differencing to remove a trend
- Seasonal differencing to remove seasonality
- Standardize to center
- Normalize to rescale
- Power Transform to make normal

This large amount of systematic searching can be slow to execute. Some ideas to speed up the evaluation of models include:

- Use multiple machines in parallel via cloud hardware (such as Amazon EC2)
- Reduce the size of the train or test dataset to make the evaluation process faster
- Use a more coarse grid of hyperparameters and circle back if you have time later
- Perhaps do not refit a model for each step in walk-forward validation


#### Step 4: Finalize Model

At the end of the previous time step, you know whether your time series is predictable. If it is predictable, you will have a list of the top 5 to 10 candidate models that are skillful on the problem. You can pick one or multiple models and finalize them. 

This involves training a new final model on all available historical data (train and test). The model is ready for use; for example:

- Make a prediction for the future.
- Save the model to file for later use in making predictions.
- Incorporate the model into software for making predictions.

---

## 3. Simple and Classical Forecasting Methods


Establishing a baseline is essential on any time series forecasting problem. A baseline in performance gives you an idea of how well all other models will actually perform on your problem.

### Simple Forecasting Methods

Three properties of a good technique for making a naive forecast are:

- *Simple*: A method that requires little or no training or intelligence.
- *Fast*: A method that is fast to implement and computationally trivial to make a prediction.
- *Repeatable*: A method that is deterministic, meaning that it produces an expected output given the same input.

#### Forecast Strategies

Simple forecast strategies are those that assume little or nothing about the nature of the forecast problem and are fast to implement and calculate. If a model can perform better than the performance of a simple forecast strategy, then it can be said to be skillful. There are two main themes to simple forecast strategies; they are:

- *Naive*, or using observations values directly.
- *Average*, or using a statistic calculated on previous observations.

1. Naive Forecasting Strategy

A naive forecast involves using the previous observation directly as the forecast without any change. It is often called the persistence forecast as the prior observation is persisted.

For example, given the series:

`[1, 2, 3, 4, 5, 6, 7, 8, 9]`

We could persist the last observation (relative index -1) as the value 9 or persist the second last prior observation (relative index -2) as 8, and so on.

2. Average Forecast Strategy

One step above the naive forecast is the strategy of averaging prior values. All prior observations are collected and averaged, either using the mean or the median, with no other treatment to the data.

For example, given the series:

`[1, 2, 3, 4, 5, 6, 7, 8, 9]`

We could average the last one observation (9), the last two observations (8, 9), and so on. In the case of seasonal data, we may want to average the last n-prior observations at the same time in the cycle as the time that is being forecasted. 

### Autoregressive Methods

- Autoregressive Integrated Moving Average, or ARIMA, is one of the most widely used forecasting methods for univariate time series data forecasting.
- An extension to ARIMA that supports the direct modeling of the seasonal component of the series is called SARIMA.

#### Autoregressive Integrated Moving Average Model(ARIMA)

An ARIMA model is a class of statistical models for analyzing and forecasting time series data. It explicitly caters to a suite of standard structures in time series data, and as such provides a simple yet powerful method for making skillful time series forecasts.

ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally adjusted via methods such as seasonal differencing.

- **AR(p): Autoregression** - A model that uses the dependent relationship between an observation and some number of lagged observations.
- **I(d): Integrated** - The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
- **MA(q): Moving Average** - A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

#### Seasonal Autoregressive Integrated Moving Average Model(SARIMA)

Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.

Configuring a SARIMA requires selecting hyperparameters for both the trend and seasonal elements of the series.

**Trend Elements**

There are three trend elements that require configuration. They are the same as the ARIMA model; specifically:

- p: Trend autoregression order
- d: Trend difference order
- q: Trend moving average order

**Seasonal Elements**

There are four seasonal elements that are not part of ARIMA that must be configured; they are:

- P: Seasonal autoregressive order.
- D: Seasonal difference order.
- Q: Seasonal moving average order.
- m: The number of time steps for a single seasonal period.

### Exponential Smoothing Methods

Exponential smoothing is a time series forecasting method for univariate data that can be extended to support data with a systematic trend or seasonal component. It is a powerful forecasting method that may be used as an alternative to the popular Box-Jenkins ARIMA family of methods.

#### What Is Exponential Smoothing?

Time series methods like the Box-Jenkins ARIMA family of methods develop a model where the prediction is a weighted linear sum of recent past observations or lags. Exponential smoothing forecasting methods are similar in that a prediction is a weighted sum of past observations, but the model explicitly uses an exponentially decreasing weight for past observations. Specifically, past observations are weighted with a geometrically decreasing ratio.

There are three main types of exponential smoothing time series forecasting methods.

1. Single Exponential Smoothing

- Single Exponential Smoothing, SES for short, also called Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality.
- It requires a single parameter, called alpha (a or α), also called the smoothing factor or smoothing coefficient.
- This parameter controls the rate at which the influence of the observations at prior time steps decay exponentially. Alpha is often set to a value between 0 and 1.
- Large values mean that the model pays attention mainly to the most recent past observations, whereas smaller values mean more of the history is taken into account when making a prediction.

Hyperparameters:
- *Alpha (α)*: Smoothing factor for the level.

2. Double Exponential Smoothing

- Double Exponential Smoothing is an extension to Exponential Smoothing that explicitly adds support for trends in the univariate time series. 
- In addition to the alpha parameter for controlling smoothing factor for the level, an additional smoothing factor is added to control the decay of the influence of the change in trend called beta (b or β).
- The method supports trends that change in different ways: an additive and a multiplicative, depending on whether the trend is linear or exponential respectively.
- Double Exponential Smoothing with an additive trend is classically referred to as Holt’s linear trend model.
- *Additive Trend*: Double Exponential Smoothing with a linear trend.
- *Multiplicative Trend*: Double Exponential Smoothing with an exponential trend.

Hyperparameters:

- Alpha (α): Smoothing factor for the level
- Beta (β): Smoothing factor for the trend
- Trend Type: Additive or multiplicative
- Dampen Type: Additive or multiplicative
- Phi (φ): Damping coefficient

3. Triple Exponential Smoothing

Triple Exponential Smoothing is an extension of Exponential Smoothing that explicitly adds support for seasonality to the univariate time series. This method is sometimes called Holt- Winters Exponential Smoothing.

As with the trend, the seasonality may be modeled as either an additive or multiplicative process for a linear or exponential change in the seasonality.

- *Additive Seasonality*: Triple Exponential Smoothing with a linear seasonality.
- *Multiplicative Seasonality:* Triple Exponential Smoothing with an exponential season-
ality.

Hyperparameters:

- Alpha (α): Smoothing factor for the level
- Beta (β): Smoothing factor for the trend
- Trend Type: Additive or multiplicative
- Dampen Type: Additive or multiplicative
- Phi (φ): Damping coefficient
- Gamma (γ): Smoothing factor for the seasonality
- Seasonality Type: Additive or multiplicative
- Period: Time steps in seasonal period

---

## Time Series to Supervised Learning Problem

Time series forecasting can be framed as a supervised learning problem. This re-framing of your time series data allows you access to the suite of standard linear and nonlinear machine learning algorithms on your problem.

### Supervised Machine Learning

The majority of practical machine learning uses supervised learning. Supervised learning is where you have input variables (X) and an output variable (y) and you use an algorithm to learn the mapping function from the input to the output.

$$Y =f(X)$$

Below is a contrived example of a supervised learning dataset where each row is an observation comprised of one input variable (X) and one output variable to be predicted (y).

                                                            | X | y   |
                                                            |---|-----|
                                                            | 5 | 0.9 |
                                                            | 4 | 0.8 |
                                                            | 5 | 1.0 |
                                                            | 3 | 0.7 |
                                                            | 4 | 0.9 |

Supervised learning problems can be further grouped into regression and classification problems.

- **Classification**: A classification problem is when the output variable is a category, such as red and blue or disease and no disease.
- **Regression**: A regression problem is when the output variable is a real value, such as dollars or weight. The contrived example above is a regression problem.

### Sliding Window

Time series data can be phrased as supervised learning. Given a sequence of numbers for a time series dataset, we can restructure the data to look like a supervised learning problem.

                                                            | time | measure |
                                                            |------|---------|
                                                            | 1    | 100     |
                                                            | 2    | 110     |
                                                            | 3    | 108     |
                                                            | 4    | 115     |
                                                            | 5    | 120     |

We can restructure this time series dataset as a supervised learning problem by using the value at the previous time step to predict the value at the next time step. Re-organizing the time series dataset this way, the data would look as follows:

                                                            | X   | y   |
                                                            |-----|-----|
                                                            | ?   | 100 |
                                                            | 100 | 110 |
                                                            | 110 | 108 |
                                                            | 108 | 115 |
                                                            | 115 | 120 |
                                                            | 120 | ?   |

- We can delete 1st and last row since they have missing value before training a supervised model.
- The use of prior time steps to predict the next time step is called the sliding window method.

### Sliding Window With Multiple Variates

The number of observations recorded for a given time in a time series dataset matters. Tradi- tionally, different names are used:

- **Univariate Time Series**: These are datasets where only a single variable is observed at each time, such as temperature each hour. The example in the previous section is a univariate time series dataset.
- **Multivariate Time Series**: These are datasets where two or more variables are observed at each time.

For example suppose we have following dataset:

                                                            | time | measure1 | measure2 |
                                                            |------|----------|----------|
                                                            | 1    | 0.2      | 88       |
                                                            | 2    | 0.5      | 89       |
                                                            | 3    | 0.7      | 87       |
                                                            | 4    | 0.4      | 88       |
                                                            | 5    | 1.0      | 90       |

Let’s also assume that we are only concerned with predicting measure2. We can re-frame this time series dataset as a supervised learning problem with a window width of one.

                                                            | X1  | X2  | X3  | y   |
                                                            |-----|-----|-----|-----|
                                                            | ?   | ?   | 0.2 | 88  |
                                                            | 0.2 | 88  | 0.5 | 89  |
                                                            | 0.5 | 89  | 0.7 | 87  |
                                                            | 0.7 | 87  | 0.4 | 88  |
                                                            | 0.4 | 88  | 1.0 | 90  |
                                                            | 1.0 | 90  | ?   | ?   |

We can see that as in the univariate time series example above, we may need to remove the first and last rows in order to train our supervised learning model. 

If we need to predict both `measure1` and `measure2` for the next time step. We can transform the data as follows:

                                                            | X1  | X2  | y1  | y2  |
                                                            |-----|-----|-----|-----|
                                                            | ?   | ?   | 0.2 | 88  |
                                                            | 0.2 | 88  | 0.5 | 89  |
                                                            | 0.5 | 89  | 0.7 | 87  |
                                                            | 0.7 | 87  | 0.4 | 88  |
                                                            | 0.4 | 88  | 1.0 | 90  |
                                                            | 1.0 | 90  | ?   | ?   |

### Sliding Window With Multiple Steps

- **One-step Forecast**: This is where the next time step (t+1) is predicted.
- **Multi-step Forecast**: This is where two or more future time steps are to be predicted.

Consider this univariate time series dataset:

                                                            | time | measure |
                                                            |------|---------|
                                                            | 1    | 100     |
                                                            | 2    | 110     |
                                                            | 3    | 108     |
                                                            | 4    | 115     |
                                                            | 5    | 120     |

We can frame this time series as a two-step forecasting dataset for supervised learning with a window width of one, as follows:

                                                            | X1  | y1  | y2  |
                                                            |-----|-----|-----|
                                                            | ?   | 100 | 110 |
                                                            | 100 | 110 | 108 |
                                                            | 110 | 108 | 115 |
                                                            | 108 | 115 | 120 |
                                                            | 115 | 120 | ?   |
                                                            | 120 | ?   | ?   |

Specifically, that a supervised model only has X1 to work with in order to predict both y1 and y2. 

There are several ways to prepare the time series data for supervised machine learning algorithms.

---


## Deep Learning for Time Series Forecasting

Deep learning neural networks are able to automatically learn arbitrary complex mappings from inputs to outputs and support multiple inputs and outputs. These are powerful features that offer a lot of promise for time series forecasting, particularly on problems with complex-nonlinear dependencies, multivalent inputs, and multi-step forecasting. These features along with the capabilities of more modern neural networks may offer great promise such as the automatic feature learning provided by convolutional neural networks and the native support for sequence data in recurrent neural networks. 

Limitations with Classical Methods like ARIMA models:

- Focus on complete data: missing or corrupt data is generally unsupported.
- Focus on linear relationships: assuming a linear relationship excludes more complex joint distributions.
- Focus on fixed temporal dependence: the relationship between observations at different times, and in turn the number of lag observations provided as input, must be diagnosed and specified.
- Focus on univariate data: many real-world problems have multiple input variables.
- Focus on one-step forecasts: many real-world problems require forecasts with a long time horizon.

We will review the models available in deep learning domain. There are mainly 3 models available:

1. Multilayer Perceptrons for Time Series
2. Convolutional Neural Networks for Time Series
3. Recurrent Neural Networks for Time Series

### Multilayer Perceptrons for Time Series

Simpler neural networks such as the Multilayer Perceptron or MLP approximate a mapping function from input variables to output variables. 

This general capability is valuable for time series for a number of reasons.

- **Robust to Noise** - Neural networks are robust to noise in input data and in the mapping function and can even support learning and prediction in the presence of missing values.
- **Nonlinear** - Neural networks do not make strong assumptions about the mapping function and readily learn linear and nonlinear relationships.
- **Multivariate Inputs** - An arbitrary number of input features can be specified, providing direct support for multivariate forecasting.
- **Multi-step Forecasts** - An arbitrary number of output values can be specified, providing direct support for multi-step and even multivariate forecasting.

Limitations:

- **Fixed Inputs** - The number of lag input variables is fixed, in the same way as traditional time series forecasting methods.
- **Fixed Outputs** - The number of output variables is also fixed; although a more subtle issue, it means that for each input pattern, one output must be produced.

### Convolutional Neural Networks for Time Series

The ability of CNNs to learn and automatically extract features from raw input data can be applied to time series forecasting problems. A sequence of observations can be treated like a one-dimensional image that a CNN model can read and distill into the most salient elements.

Benefits:

- Support for multivariate input, multivariate output and learning arbitrary but complex functional relationships.

### Recurrent Neural Networks for Time Series

Recurrent neural networks like the Long Short-Term Memory network or LSTM add the explicit handling of order between observations when learning a mapping function from inputs to outputs, not offered by MLPs or CNNs. They are a type of neural network that adds native support for input data comprised of sequences of observations.

- **Native Support for Sequences** - Recurrent neural networks directly add support for input sequence data.

- **Learned Temporal Dependence** - The most relevant context of input observations to the expected output is learned and can change dynamically.


### Hybrid Models

The above capabilities of different models can also be combined, such as in the use of hybrid models like CNN-LSTMs and ConvLSTMs that seek to harness the capabilities of all three model types.

- Hybrid models efficiently combine the diverse capabilities of different architectures.

---