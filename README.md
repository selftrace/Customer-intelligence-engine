## Customer Intelligence Engine

**Customer segmentation, retention analysis & churn prediction using Python**

I wanted to build something more useful than another classification model with an accuracy score, so I worked on a customer intelligence project using the UCI online retail dataset. The idea was to take raw transaction data and turn it into something a business could actually use: understand who its customers are, see how their purchasing behavior differs, track whether they come back and identify customers who might stop buying.

The project combines RFM analysis, unsupervised learning, cohort retention, statistical testing and a churn prediction model in one workflow.

## What this project does

- **Customer segmentation:** groups customers based on how recently they purchased, how frequently they order and how much they spend.
- **Cohort analysis:** tracks customer retention month by month after their first purchase.
- **Statistical testing:** checks whether spending distributions differ across the discovered customer segments.
- **Churn prediction:** trains a logistic regression model to estimate which customers are unlikely to make another purchase during a defined 90-day period.
- **Customer-level insights:** visualizes segments and ranks test-set customers by predicted churn risk.

## Dataset

The project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), which contains transaction records from a UK-based online retail business between **December 2010 and December 2011**.

The dataset includes invoice numbers, product descriptions, quantities, invoice dates, unit prices, customer IDs and countries.

This is transaction-level data rather than a ready-made customer table, so the first step is to clean it and build the customer-level features needed for analysis.

### Data cleaning

Before calculating anything, I:

- Removed records without a customer ID or invoice date.
- Excluded cancelled invoices identified by invoice numbers starting with `C`.
- Removed transactions with non-positive quantities or unit prices.
- Converted invoice dates to datetime format and customer IDs to consistent strings.
- Calculated transaction revenue as `Quantity × UnitPrice`.

These steps leave a more consistent dataset for analyzing completed purchases. They also mean the analysis focuses on positive-value sales, rather than returns and cancellations.

## 1. RFM analysis & customer segmentation

RFM stands for **Recency, frequency and monetary value**. It is a simple way to describe purchasing behavior without needing pre-existing customer labels.

For each customer, I calculate:

| Feature | Calculation |
-------------------------
| Recency | Days since the customer's most recent purchase |
| Frequency | Number of distinct invoices |
| Monetary | Total sales attributed to the customer |

The reference date is based on the latest invoice date in the dataset, with recency calculated in days.

### K-means clustering

I apply a log transformation using `numpy.log1p()` to reduce the influence of highly skewed purchasing values, then standardize the features with `StandardScaler` before running K-means.

The model uses **4 clusters**, with `random_state=42` and `n_init=10` for reproducibility.

It's a starting point for exploring behavioral differences and can be tested against alternative cluster counts.

### PCA visualization

Since the clustering uses three features, I use Principal component analysis (PCA) to project the standardized RFM data into two dimensions.

The resulting Plotly scatter plot lets me explore how customers are distributed across the segments, with customer IDs and their original RFM values available on hover.

PCA is only used to visualize the data here; the K-means model clusters customers using the full three-dimensional standardized feature space.

## 2. Cohort analysis & retention

Customer segmentation shows how customers differ, but it doesn't tell us whether they continue buying over time.

For this, I group customers into cohorts based on the month of their first recorded purchase.

Each subsequent purchase is assigned a cohort age, measured in months since that first purchase. I then count unique purchasing customers within each cohort and month.

Retention is calculated as:

`Monthly cohort customers / Cohort customers in month 0 × 100`

The result is an interactive plotly heatmap showing monthly retention percentages.

This makes it possible to compare customer groups that started purchasing at different times and see how retention changes as cohorts age.

One limitation: the final months of the dataset have less follow-up time, so their retention patterns should not be compared directly with older, more complete cohorts.

## 3. Statistical testing

After segmenting customers, I use the **Kruskal–Wallis H-test** from `scipy.stats` to compare their monetary values.

I chose this non-parametric test because customer spending is often highly skewed, with a small number of customers contributing a large share of revenue.

The test evaluates whether the spending distributions across the segments are consistent with the same underlying distribution.

I use a significance threshold of `p < 0.05`.

A significant result would provide evidence that at least one segment differs from the others. It would not tell us which specific pairs differ, nor prove that the clustering itself is meaningful. Follow-up pairwise tests and multiple-testing corrections would be needed for that.

## 4. Churn prediction

The last part of the project uses logistic regression to estimate customer churn risk.

Since the dataset doesn't contain an explicit churn label, I define a practical proxy based on purchasing behavior during the final 90 days of the available data.

### Target definition

- **Observation period:** transactions before the final 90-day cutoff.
- **Outcome period:** the final 90 days of the dataset.
- **Churn proxy = 1:** a customer with historical purchases before the cutoff makes no recorded purchase during the outcome period.
- **Churn proxy = 0:** a customer makes at least one recorded purchase during the outcome period.

The model uses four features calculated from the observation period:

- Recency
- Purchase frequency
- Total monetary value
- Average order value

The features are standardized, and a logistic regression model is trained with `class_weight="balanced"` to account for potential class imbalance.

### Evaluation

I use a stratified 75/25 train-test split with `random_state=42`, then evaluate the predictions using **ROC-AUC**.

ROC-AUC measures how well the model ranks customers who receive the positive label above those who do not. It does not measure probability calibration, and it should not be interpreted as accuracy.

The code also ranks customers in the held-out test set by predicted churn risk and displays the ten highest-risk customers.

**Important limitation:** this is an initial experiment, not a production-ready churn system. The target is a proxy for not returning within a particular historical window, not proof that a customer has permanently churned. The model should be validated using additional temporal cutoffs before being used for real decisions.

## Tech stack

| Library | How I use it |
--------------------------
| `pandas` | Cleaning, aggregation, RFM features and cohort tables |
| `NumPy` | Numerical transformations and feature preparation |
| `scikit-learn` | K-means, PCA, scaling, logistic regression and model evaluation |
| `SciPy` | Kruskal–Wallis statistical test |
| `Plotly Express` | Interactive customer segmentation and retention visualizations |
| `openpyxl` | Reading the Excel dataset through pandas |

## Run it locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd customer-intelligence-engine
```

### 2. Install the dependencies

```bash
pip install pandas numpy openpyxl scikit-learn scipy plotly
```

### 3. Add the dataset

Download the Online Retail dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place `online+retail.zip` in the project directory.

The current notebook code expects this path:

```python
zip_path = "/content/online+retail.zip"
```

If you're running it locally rather than in Google Colab, change that line to the path where you saved the dataset, for example:

```python
zip_path = "online+retail.zip"
```

### 4. Run the notebook

Run the cells in order in Jupyter Notebook or Google Colab. The notebook will display the customer segmentation scatter plot, retention heatmap, statistical test results, ROC-AUC score and highest-risk customers.

## Project structure

```text
customer-intelligence-engine/
├── customer_intelligence.ipynb
├── README.md
└── online+retail.zip  # local dataset; not committed to GitHub
```

## What I learned

This project gave me practice with more than just fitting a model. I worked with transaction-level data, transformed it into customer-level features, explored unsupervised learning, analyzed retention over time, used a statistical test to compare groups and built a supervised model around an explicitly defined target.

There is more to applied data science -- the output is only as useful as the assumptions behind it. Cluster labels need interpretation, a significant p-value doesn't automatically mean business value and a churn model is only meaningful when its target and evaluation setup reflect the question being asked.

## Possible next steps, WIP

- Compare different values of K using silhouette scores and cluster stability.
- Profile each segment and give it a descriptive label based on its actual purchasing behavior.
- Add pairwise statistical tests with multiple-comparison correction.
- Evaluate churn prediction using multiple time-based cutoffs rather than a single random holdout.
- Compare logistic regression with a simple baseline and additional models.
- Add an interactive customer explorer or dashboard.

---

**Built with Python, pandas, NumPy, scikit-learn, SciPy and Plotly.**
