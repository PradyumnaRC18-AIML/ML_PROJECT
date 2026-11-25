# Forecasting Air Pollution in Bidhannagar using Machine Learning

**CA-3 | Introduction to Machine Learning | PCC-AIML502**

**Techno Main Salt Lake | Semester 5**

---

## 1. Project Overview

Air pollution is a major environmental concern, particularly in densely populated urban areas. One of the important indicators used to monitor air quality is **PM2.5 (Particulate Matter 2.5)**.

PM2.5 refers to fine airborne particles with a diameter of 2.5 micrometres or smaller. Due to their extremely small size, these particles can penetrate deep into the respiratory system.

Monitoring current pollution levels is important, but being able to estimate pollution levels in advance can be equally useful.

This project explores whether historical PM2.5 readings can help predict the next day's average PM2.5 concentration in Bidhannagar, West Bengal.

We use historical hourly pollution records, transform them into daily observations, engineer time-based features, and compare three forecasting approaches.

### Research Question

> Can yesterday's readings help forecast tomorrow's PM2.5 in Bidhannagar?

### Our Objective

At the end of a particular day, represented as day **D**, we want to predict the average PM2.5 concentration for the next calendar day, **D+1**, using only information available up to day D.

For example:

- Today's average PM2.5 = 120 µg/m³.
- The model receives today's and earlier pollution readings.
- The model predicts tomorrow's average PM2.5.
- The predicted value is compared with tomorrow's actual average.

The objective is to investigate whether machine learning models can improve upon a simple persistence-based forecast.

---

## 2. Project Information

| Attribute | Details |
|---|---|
| Institution | Techno Main Salt Lake |
| Department | Computer Science and Engineering (AIML) |
| Course | Introduction to Machine Learning |
| Course Code | PCC-AIML502 |
| Semester | 5 |
| Assessment | CA-3 |
| Project | Project 1 — Air Quality Forecasting |
| Location | Bidhannagar, West Bengal |
| Target Variable | Daily Mean PM2.5 |
| Problem Type | Time-Series Regression / Forecasting |

### Team Members

1. Pradyumna Roy Choudhury
2. Param
3. Paramita
4. Pameila
5. Talha
6. Prahlad

*Roll numbers and individual contributions can be added once the team finalizes them.*

---

## 3. Dataset

The project uses hourly air-quality observations from the Bidhannagar WBPCB monitoring station.

**Data source:** Central Pollution Control Board (CPCB), published through OpenCity.

The original dataset contains hourly PM2.5 observations arranged in an unusual monthly-block CSV structure. It is therefore not a conventional flat CSV file that can simply be loaded using `pandas.read_csv()` with default settings.

A custom parsing procedure is used in the notebook to extract the year, month, day, hour, and PM2.5 values.

### Dataset Coverage

The parsed data used in our current experiment covers:

**August 2019 – December 2023**

| Dataset information | Count |
|---|---:|
| Parsed hourly observations | 38,280 |
| Valid hourly readings | 35,835 |
| Missing hourly readings | 2,445 |
| Calendar days after reindexing | 1,614 |
| Valid daily means | 1,484 |
| Usable modelling rows | 1,243 |

These counts describe different stages of data preparation. They should not be interpreted as interchangeable sample sizes.

**Important:** Although the source description refers to records listed for 2017–2023, the parsed dataset used in our current notebook begins in August 2019. Our analysis reports the actual available coverage rather than assuming that all advertised years are present.

---

## 4. Understanding the Data Preprocessing

Before building a forecasting model, the hourly observations need to be converted into a reliable daily dataset.

Our preprocessing pipeline follows these steps:

### Step 1: Parse the CSV

The original CSV contains monthly sections, year markers, hourly column headings, and daily readings.

A custom parser extracts the hourly observations and converts them into a structured dataset.

### Step 2: Extract Timestamps

Each valid reading is associated with a date and hour.

This allows us to group observations by calendar day.

### Step 3: Check Duplicate Readings

Duplicate timestamps are checked before aggregation.

The current dataset contains zero duplicate timestamps.

### Step 4: Check Valid Hourly Readings

A day should have sufficient hourly observations before its daily average is calculated.

We use a minimum completeness threshold of:

**18 valid hourly readings per day.**

### Step 5: Calculate Daily Mean PM2.5

For each date that satisfies the completeness condition, the average of its valid hourly readings is calculated.

```text
Daily Mean PM2.5 = Sum of valid hourly PM2.5 readings / Number of valid hours
```

### Step 6: Reindex the Daily Calendar

The daily dataset is reindexed to include every calendar date in the observed period.

Days that do not satisfy the 18-hour rule remain missing.

We do not artificially fill incomplete daily observations.

**Why is this important?**

If incomplete days were treated as reliable daily averages, they could introduce misleading information into the forecasting process.

---

## 5. Feature Engineering

Feature engineering means creating useful input variables from the existing dataset so that a machine learning model can learn patterns more effectively.

Our objective is to predict the PM2.5 mean of day D+1 using information available up to day D.

The notebook constructs the following features:

| Feature | Explanation |
|---|---|
| `pm25_today` | Average PM2.5 on the current day D |
| `pm25_yesterday` | Average PM2.5 on day D-1 |
| `pm25_2days_ago` | Average PM2.5 on day D-2 |
| `pm25_roll7` | Average PM2.5 over the seven-day period ending on D |
| `month_sin` | Cyclical representation of the calendar month |
| `month_cos` | Cyclical representation of the calendar month |

### What is a Seven-Day Rolling Mean?

Instead of considering only one day's pollution level, we can calculate the average pollution over the previous seven calendar days.

For day D:

```text
RollingMean_7(D) = (PM2.5(D - 6) + PM2.5(D - 5) + ... + PM2.5(D)) / 7
```

This provides information about the recent pollution trend.

The rolling mean requires all seven daily observations to be valid in the current implementation.

### What is the Target Variable?

The target is:

```text
Target = PM2.5(D + 1)
```

In simple terms, the model uses today's and earlier readings to predict tomorrow's average PM2.5.

### Avoiding Data Leakage

Data leakage happens when information that would not have been available at prediction time accidentally enters the model.

For example, using tomorrow's actual PM2.5 while predicting tomorrow's PM2.5 would make the evaluation invalid.

Our features use information available only up to day D. The target is the next calendar day's daily mean.

Rows without the required valid features or target are excluded from modelling.

---

## 6. Forecasting Approaches

We compare three approaches.

### 6.1 Persistence Baseline

Persistence is our simplest forecasting method.

It assumes that tomorrow's pollution level will be the same as today's.

```text
Predicted PM2.5(D + 1) = PM2.5(D)
```

For example, if today's average PM2.5 is 120 µg/m³, persistence predicts tomorrow's average as 120 µg/m³.

**Does persistence use machine learning?**

No. It is a baseline forecasting rule.

We use it to determine whether more complex models actually provide an improvement.

### 6.2 Ridge Regression

Ridge Regression is a regularized version of linear regression.

It learns a linear relationship between the input features and the target variable.

It also introduces an L2 penalty to discourage excessively large model coefficients.

This can help when input features are correlated with one another, such as different lagged PM2.5 readings.

The notebook uses feature scaling and selects the regularization strength using the validation period.

### 6.3 Random Forest Regressor

Random Forest is an ensemble learning algorithm.

It creates multiple decision trees and combines their predictions to produce the final regression output.

Unlike a simple linear model, Random Forest can learn certain non-linear relationships between the input features and the target.

The notebook tests different tree-depth and minimum-leaf-size settings using the validation period.

---

## 7. Model Evaluation Strategy

Because this project deals with time-dependent observations, randomly shuffling the data into training and testing sets is not appropriate.

A random split could allow the model to train on observations from a later period while being evaluated on earlier observations.

Instead, we use a **chronological split**.

| Dataset | Forecast Date Range | Purpose |
|---|---|---|
| Training | Through 31 December 2021 | Learn model parameters |
| Validation | 1 January – 31 December 2022 | Select model settings |
| Testing | 1 January 2023 onwards | Final performance evaluation |

The validation dataset is used to select model settings.

The selected machine learning models are then refitted using the combined training and validation data before final evaluation on the test period.

The test period is kept separate from model selection.

### Evaluation Metric: Mean Absolute Error (MAE)

We use Mean Absolute Error to measure forecasting performance.

```text
MAE = Sum of |y_i - y_hat_i| / n = Sum of absolute errors / Number of predictions
```

Where:

- `y_i` represents the actual PM2.5 value.
- `y_hat_i` represents the predicted PM2.5 value.
- `n` represents the number of forecasts.

MAE tells us the average absolute difference between actual and predicted values.

Its unit is µg/m³.

**Lower MAE indicates better forecasting performance.**

---

## 8. Experimental Results

The final evaluation compares all three forecasting approaches on identical test dates.

### Full Test Period — 2023

| Model | MAE (µg/m³) |
|---|---:|
| Persistence | 22.66 |
| Ridge Regression | **21.84** |
| Random Forest | 22.67 |

Number of valid forecasts: **319**

### December–February Subset

| Model | MAE (µg/m³) |
|---|---:|
| Persistence | 39.18 |
| Ridge Regression | **36.67** |
| Random Forest | 37.63 |

Number of valid forecasts: **88**

### Interpretation

Ridge Regression achieved the lowest MAE in both final test comparisons.

Compared with persistence:

- Full test MAE improved from 22.66 to 21.84 µg/m³.
- December–February MAE improved from 39.18 to 36.67 µg/m³.

This represents an approximate 3.6% reduction in full-test MAE and a 6.4% reduction in the December–February subset.

However, an important observation is that persistence performed better than both machine learning models during validation in 2022.

| Model | Validation MAE (µg/m³) |
|---|---:|
| Persistence | **16.29** |
| Ridge Regression | 16.85 |
| Random Forest | 17.01 |

This demonstrates that machine learning models do not automatically outperform simple baselines in every period.

We report the validation and test results separately rather than hiding this difference.

---

## 9. Error Analysis

A model's average performance does not explain every prediction.

Some days may have relatively small errors, while others may have substantial differences between actual and predicted pollution levels.

Our notebook investigates the three largest absolute Ridge prediction errors.

| Forecast Date | Actual PM2.5 | Predicted PM2.5 | Absolute Error |
|---|---:|---:|---:|
| 7 January 2023 | 344.3 | 176.3 | 168.0 |
| 14 January 2023 | 100.1 | 215.5 | 115.4 |
| 19 March 2023 | 41.0 | 151.8 | 110.8 |

All PM2.5 values and errors are in µg/m³.

These examples demonstrate that historical readings alone may not capture abrupt changes in pollution.

For example, a sudden increase or decrease can be difficult for a model to predict using only recent PM2.5 history.

The surrounding days are also inspected in the notebook to understand these errors.

---

## 10. Limitations

Our project has several limitations.

**1. Single Monitoring Station**

The analysis focuses on the Bidhannagar monitoring station. Results should not automatically be generalized to other locations.

**2. Historical PM2.5 Features Only**

The current forecasting models primarily use historical PM2.5 readings and calendar-related features.

Weather variables such as wind speed, humidity, and temperature are not included.

**3. Missing and Incomplete Observations**

Some days do not satisfy the minimum 18-hour completeness rule. These days remain missing and cannot be used as ordinary daily observations.

**4. Sudden Pollution Changes**

Abrupt pollution spikes and clearances are difficult to predict using historical pollution readings alone.

**5. Dataset Coverage and Metadata**

The parsed dataset used in this experiment covers August 2019 to December 2023. The original CSV also does not independently provide all station and measurement metadata in its contents.

---

## 11. Future Scope

Possible improvements include:

- Integrating weather information such as wind speed, humidity, and temperature.
- Exploring additional regression and ensemble models.
- Investigating longer forecasting horizons.
- Comparing multiple air-quality monitoring stations.
- Developing a simple dashboard to display predicted PM2.5 levels.

These are possible future improvements and are not claimed as completed features of the current implementation.

---

## 12. Repository Structure

The repository may be organized as follows:

```text
CA3-Air-Quality-Forecasting/
│
├── README.md
│
├── notebook/
│   └── CA3_Air_Quality_Forecasting_Simple.ipynb
│
├── data/
│   └── 39247c0a-3540-480f-8cde-7b54cfa561d7.csv
│
├── report/
│   ├── CA3_Air_Quality_Forecasting_Report.docx
│   └── CA3_Air_Quality_Forecasting_Report.pdf
│
├── presentation/
│   └── CA3_Air_Quality_Forecasting_Presentation.pptx
│
└── figures/
    └── exported_charts/
```

This is the suggested organization. The actual paths should match the final repository.

---

## 13. How to Run the Project

### Requirements

The project uses Python and Jupyter Notebook.

Required libraries:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- IPython

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd CA3-Air-Quality-Forecasting
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Step 3: Place the Dataset Correctly

The notebook currently uses the filename:

```text
39247c0a-3540-480f-8cde-7b54cfa561d7.csv
```

Make sure the CSV is accessible at the path expected by the notebook.

If the dataset is stored in a separate `data/` directory, the notebook's data-file path must point to that location.

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
CA3_Air_Quality_Forecasting_Simple.ipynb
```

### Step 5: Execute the Notebook

Run the cells sequentially from top to bottom.

The notebook performs:

1. Data parsing.
2. Data exploration.
3. Data cleaning.
4. Daily aggregation.
5. Feature engineering.
6. Chronological splitting.
7. Baseline forecasting.
8. Ridge Regression.
9. Random Forest.
10. Model comparison.
11. Visualization.
12. Error analysis.

**Important:** Run the notebook from a fresh kernel before using its outputs in the final report or presentation.

---

## 14. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Development and experimentation |
| NumPy | Numerical operations |
| Pandas | Data processing and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models and evaluation |
| Git and GitHub | Version control and collaboration |

---

## 15. Team Contributions

The following table should be completed by the team before final submission.

| Team Member | Contribution |
|---|---|
| Pradyumna Roy Choudhury | To be finalized |
| Param | To be finalized |
| Paramita | To be finalized |
| Pameila | To be finalized |
| Talha | To be finalized |
| Prahlad | To be finalized |

Contributions should reflect the actual work performed, such as data analysis, preprocessing, model development, visualization, documentation, testing, and presentation preparation.

---

## 16. Conclusion

This project investigates whether historical PM2.5 readings can help forecast the next day's average pollution level in Bidhannagar.

We constructed a daily dataset from hourly observations, applied an 18-valid-hour completeness rule, engineered lagged and rolling features, and compared three forecasting approaches.

On the 2023 test period, Ridge Regression achieved the lowest MAE of 21.84 µg/m³, compared with 22.66 µg/m³ for persistence and 22.67 µg/m³ for Random Forest.

The results suggest that historical PM2.5 readings contain useful information for next-day forecasting. However, the improvement over persistence is modest, and sudden pollution changes remain difficult to predict.

The experiment also highlights why chronological evaluation and comparison against a simple baseline are important when building time-series forecasting models.

---

**Project:** Forecasting Air Pollution in Bidhannagar  
**Course:** PCC-AIML502 — Introduction to Machine Learning  
**Institution:** Techno Main Salt Lake

**Developed as part of the CA-3 Machine Learning Group Project.**