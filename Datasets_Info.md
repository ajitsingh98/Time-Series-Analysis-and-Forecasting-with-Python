# Datasets Summary

### 1. `airline_passengers.csv`
- **Description**: This dataset contains monthly totals of international airline passengers from 1949 to 1960.
- **Shape**: 144 rows, 2 columns.
- **Columns**: 
  - `Month`: The month of the record (YYYY-MM format).
  - `Thousands of Passengers`: The number of passengers (in thousands).

### 2. `Alcohol_Sales.csv`
- **Description**: This dataset represents monthly alcohol sales.
- **Shape**: 325 rows, 2 columns.
- **Columns**:
  - `DATE`: The date of the record (YYYY-MM-DD format).
  - `S4248SM144NCEN`: The sales figure (numeric).

### 3. `DailyTotalFemaleBirths.csv`
- **Description**: Daily total female births in California in 1959.
- **Shape**: 365 rows, 2 columns.
- **Columns**:
  - `Date`: The date of the record (MM/DD/YYYY format).
  - `Births`: The number of births (numeric).

### 4. `RestaurantVisitors.csv`
- **Description**: This dataset tracks daily visitors to four different restaurants, along with whether the day was a holiday.
- **Shape**: 517 rows, 9 columns.
- **Columns**:
  - `date`: The date of the record (MM/DD/YYYY format).
  - `weekday`: The day of the week.
  - `holiday`: Indicator if the day is a holiday (1 for holiday, 0 otherwise).
  - `holiday_name`: Name of the holiday, if applicable.
  - `rest1` to `rest4`: Visitor count for each restaurant (numeric).
  - `total`: Total visitors across all restaurants (numeric).


Each dataset can be used to illustrate different techniques and models in time series analysis and forecasting, ranging from basic methods to more advanced deep learning approaches. The variety in these datasets (from sales data to daily counts) provides a comprehensive resource for demonstrating different time series scenarios and methods.