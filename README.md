# Background
**Name:** SPC

**Description:** 

A typical school of equity valuation considers stock price as the sum of present value of all future dividends. One way to measure the present value of future dividend is using the underlying's put and call options at different maturities.

SPC is the slope when we regress present value of dividend payments (derived from implied volatilities of puts and calls) on the options' maturity. This indicator is the average SPC of the nearest full week. According to empirical findings, when SPC increases, the discount rate for future dividend decreases, suggesting lower future index return. Therefore, this indicator reflects option traders' collective sentiment and can be served as a barometer for future market movements. We suggest the following strategy: long the HSI when this week's average SPC is lower than previous week, otherwise don't hold any position.

This strategy has been tested in a period of 15 years (2006 to 2021). As compared with a simple long strategy in HSI, it delivers better return with lower risk. Performance statistics

are shown below:

- Cumulative Return: 173.39% versus 57.56% (long HSI)

- Maximum Drawdown: 34.17% versus 65.18% (long HSI)

- Sharpe ratio: 0.41 versus 0.12 (long HSI)

- Volatility: 16.11% versus 23.41% (long HSI)


**SPC Formula**
- put-call parity: $P- C = K e^{-r_t T}$ -S_t + PV(Dividend)$
- theorectical put-call ratio: $\frac{P-C}{C}=\frac{P}{C} - 1$ where $P-C$ is approximately $PV(Dividend)$ for ATM options
- actual put-call ratio: 
  $\frac{V^{BS}_{PUT}(S_t/K=1, T, \sigma_t^{ATM put})}{V^{BS}_{CALL}(S_t/K=1, T, \sigma_t^{ATM call})} - 1$ for $T$ ranging from 1/12 to 1
- slope $spc$: slope of regressing put-call ratio over $T$
  
- Back-solving for implied vol using the Newton's method: $\sigma_{n+1} = \sigma_{n} - \frac{BS(\sigma_n)-P}{v(\sigma_n)}$ where 
$P$ is the option price and $v(\sigma_n)$ is vega. 




# Files
1. weekly update of SPC: `main.py`, `option.py` (implementation class) and  `utils.py`(functions for the class)
2. backtest results (in the *backtests* folder): `main.py`, `strat.py` (strategy classes) and `backtest.ipynb` (visualization)
3. reference (in the *backtests* folder): SPC theories, Refinitiv's implied vol computation methodology, formulas for Black-Scholes Models
4. reference online: [Computing implied volatility](https://quant.stackexchange.com/questions/7761/a-simple-formula-for-calculating-implied-volatility)

# Calculation
## Data Inputs
- HIBOR quotes (1M to 12M)
- HSI close price
- ATM option close price (maturity ranging from 1M to 12M)

## Procedures
- Get HSI close price ($S_t$) and mid-price (average of OHLC) to determine the ticker of ATM 
options
- Construct yield curve using HIBOR quotes (cubic interpolation)
- Access option settlement prices from Refinitiv API
- Compute dividend yield implied from put-call parity using the settlement price of set 1 and set 2
$$q_1 = \frac{1}{T} ln(\frac{C_{1,t}-P_{1,t}+K e^{rT}}{S_t})$$

$$q_2 = \frac{1}{T} ln(\frac{C_{2,t}-P_{2,t}+K e^{rT}}{S_t})$$

$$q = \frac{1}{2}(q_1 + q_2)$$ 

- Solve for implied volatility of the option prices
- Compute put-call ratio of each maturity
- Compute spc of each day

# Datasets
## MySQL `HSIVolCurve` table

| Column Name      | Description |
| ----------- | ----------- | 
| date | pricing date of the options |
| type | option type ("c" or "p") |
| 1M, 2M, ...,  12M | implied volatility on that day for a option expiring in xM later|


**Specification:**
- data type: "type" is of type `char(1)`
  
> ALTER TABLE HSIVolCurve MODIFY type char(1);  
- unique keys: unique pair of "date" and "type"

> ALTER TABLE HSIVolCurve ADD CONSTRAINT date_option UNIQUE (date, type);

## MySQL `DailySPC` table
| Column Name      | Description |
| ----------- | ----------- | 
| date | pricing date of the options |
| spc | value of spc (float)|

**Specification**
- unique key: date

## MongoDB `WeeklySPC` table

| Column Name      | Description |
| ----------- | ----------- | 
| date | Friday of the calculation week |
| ratio | average spc of the given week|

**Specification**
- unique key: date
