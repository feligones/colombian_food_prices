# Time Series Forecasting Model
The forecasting method consists of a model that uses historic price information from a specific product and group characteristics. The methodology is based on Ryan Holbrook's great [Kaggle time series forecasting course](https://www.kaggle.com/learn/time-series).

The modeling process is shown in the following diagram:

![modelflow](/images/col_food_prices_modelflow.png)


## 2-Step Forecasting Method
### First-Step: Deterministic Model
Each raw prices time series is de-trended and de-seasonalized by fitting a time deterministic model using Linear Regression.
Here the inputs are a bias term, seasonal dummies for each-but-one month and 2nd order time deterministic features.

$$P_{it} = \beta_{i0} + \Theta_i*sea\_dum_t + \alpha_{i}t  + \gamma_{i} t^2$$

Following [Hyndman, R.J., & Athanasopoulos, G. (2021)](https://otexts.com/fpp3/stlfeatures.html), STL decomposition allows us to measure the strenght of trend and seasonality in each time series and, therefore, determine if this components should be controlled for. Recall that the decomposition is written as:

$$P_t = T_t + S_t + R_t$$

Where $P_t$ is the price time series, $T_t$ is the trend component, $S_t$ is the seasonal component and $R_t$ the remainder.

For strongly trended data, the seasonally adjusted data should have much more variation than the remainder component. If this is true, the trend component is controlled for. Therefore:

$$
\alpha_{i} = 
\begin{cases}
\alpha_{i}, if \max(0, 1 - \frac{Var(R_t)}{Var(R_t+T_t)}) > 0.5\\
0, otherwise
\end{cases}
$$

$$
\gamma_{i} = 
\begin{cases}
\gamma_{i}, if \max(0, 1 - \frac{Var(R_t)}{Var(R_t+T_t)}) > 0.5\\
0, otherwise
\end{cases}
$$

The same logic applies for the seasonal component, but with respect to the detrended data rather than the seasonally adjusted data:

$$
\Theta_{i} = 
\begin{cases}
\Theta_{i}, if \max(0, 1 - \frac{Var(R_t)}{Var(R_t+S_t)}) > 0.5\\
0, otherwise
\end{cases}
$$

The residuals of the fitted deterministic model ($P_{it} - \hat{P_{it}}$) are saved as well as the (partial) out-of-sample prediction ($P_{it+1}$).

### Second-Step: Residual Model
The 2nd-stage model uses the residuals of all the time series fitted separatelly in the 1st-stage. The residuals in $t$ are the model outputs and the inputs are the residuals in $t-1$, $t-2$ and $t-3$, the standard deviation of these lags and a dummy feature associated with the time series' product group. 

Any supervised learning algorithm would work, since we have *X*s and a *y*. We chose sklearn's Linear Regression for simplicity.

The forecast of the model are the predicted residuals for all the price series in $t+1$.

## Final Prediction
The final prediction is the sum of the out-of-sample deteministic (1st-stage) prediction and the residual (2nd-stage) prediction. One way to put it is that the final prediction is the sum of the predicted trend and seasonality in $t+1$ given by the deterministic model, and the remainder in the same period given by the residual model.

The forecast intervals (at a 95% confidence level) are given by:
$$P_{T+h|T} \pm 1.96 \hat{\sigma_h}$$

This assumes that the distribution of future errors is normal. We compute $\hat{\sigma_h}$ in a naive way by estimating the standard deviation of the residuals of the 1st stage + 2nd stage fitted values.

## Evaluation Metrics
We used Mean Absolute Percentage Error (MAPE) as the error metric to evaluate prediction performance. Because we are comparing predictions of hundreds of time series with different scales, this metric allows us to scale the error (true - pred) for each time series and then average it. We also include other statistics such as median and percentiles to get a broader understanding of the aggregate performance.

For a Test Set including the real prices of December 2022, the MAPE of the predictions of the model trained with data until November 2022 has the following statistics:

| Statistic |  |
| --- | ----------- |
| Mean | 12.7% |
| STD | 15.5% |
| Median | 0.071% |
| Min | 0.0% |
| P25 | 2.6% |
| P75 | 16.8% |
| Max | 175.2% |