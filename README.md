# GPU Data Analysis Project: Probability and Statistics

This project performs descriptive and inferential statistical analysis on GPU
specifications from `data/All_GPUs.csv`, focusing on two quantitative variables:

- `Max_Power` (W)
- `Release_Price` (USD)

The primary goal is to describe the sample and study how hardware attributes
such as memory, bandwidth, clock speed, release year, and manufacturer relate
to power and release price. Regression prediction is an optional extension,
not the main analysis.

## Project overview

The main workflow combines data cleaning, descriptive statistics, exploratory
visualization, confidence intervals, correlation tests, and group comparisons.
It is designed to answer questions such as:

- Which GPU features are most strongly associated with power consumption?
- Which features best explain launch price?
- Do power and price distributions differ between manufacturers?

An optional regression extension then examines how accurately `Max_Power` and
`Release_Price` can be predicted.

## What the project does

The workflow includes the following steps:

1. Read and validate the raw CSV dataset.
2. Clean the data by normalizing numeric values, handling missing entries, and converting textual fields into usable formats.
3. Standardize units where needed, such as memory bandwidth conversion.
4. Remove or treat unreleased GPU entries appropriately for price prediction.
5. Export dataset/variable metadata and a reproducible cleaning audit.
6. Compute descriptive statistics, confidence intervals, corrected correlation tests, and a non-parametric group comparison.
7. Generate numbered table images and figures for the report.
8. Export the primary statistical results to `results/statistics/`.
9. As an optional extension, train and evaluate regression models.
10. Export extension results separately to `results/regression_extension/`.

## Dataset description

The input dataset is stored in `data/All_GPUs.csv` and contains GPU-level information such as:

- `Name`
- `Manufacturer`
- `Release_Date`
- `Release_Year`
- `Memory`
- `Memory_Bandwidth`
- `Core_Speed`
- `Max_Power`
- `Release_Price`
- other GPU specification fields

The data includes both numeric and text-based entries, so the project performs conversions and cleaning before modeling. Some rows contain values such as `not released`, which are handled separately because they do not represent an observed launch price.

In total, there are 556 GPUs with original prices available for the price prediction task. Among them, 11 entries labeled `[not released]` are excluded from the price regression task.

## Optional regression extension

Regression is supplementary to the statistical report. It is a supervised
learning task for predicting continuous numeric values rather than class labels.

In this context:

- `Max_Power` is a continuous target variable measured in watts.
- `Release_Price` is a continuous target variable measured in USD.
- The model learns from historical GPU features and predicts a numeric value for each new observation.

The project compares multiple regression approaches, including:

- Ridge Regression
- Random Forest Regression

The best-performing model is selected using 5-fold cross-validation on the training set, then evaluated once on the 20% test set.

## Project structure

```text
code_Statistics/
├── data/
│   └── All_GPUs.csv           # Raw dataset
├── src/
│   ├── config.py              # Input variables and project task settings
│   ├── cleaning.py            # Read CSV, normalize units, handle unreleased prices
│   ├── statistics.py          # Descriptive statistics and correlation analysis
│   ├── visualization.py       # Plot generation
│   ├── regression.py          # Preprocessing, Ridge, and Random Forest models
│   └── analysis.py            # Train/test split, evaluation, and output export
├── results/
│   ├── statistics/           # Main numbered tables, figures, CSVs, summary
│   └── regression_extension/ # Optional prediction outputs
├── main.py                    # Entry point
├── README.md                  # Project documentation
├── requirements.txt           # Required libraries
└── .gitignore                 # Git ignore rules
```

## Run the program

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

You can also specify a different CSV source and output directory:

```bash
python main.py --data /path/to/All_GPUs.csv --output /path/to/output
```

The program writes the main report material to `results/statistics/`. Table
images use names such as `Table_01_...png`, while plots use names such as
`Figure_01_...png`; every image also contains its number and title.

Prediction outputs are kept separately in `results/regression_extension/` and
are labeled `Extension Figure`.

The generated report-ready images include:

- dataset context, variable roles/types, and observation/missing counts;
- a data-cleaning and unit-conversion audit;
- numeric and categorical descriptive-statistics tables;
- missing-data, distribution, boxplot, time-trend, correlation, and scatter plots;
- inferential tables with a 95% mean confidence interval, Pearson confidence intervals and Bonferroni-adjusted p-values, plus a Kruskal-Wallis manufacturer test and effect size.

The optional extension folder contains cross-validation/test model comparisons
and hold-out residual diagnostics.

## Important notes

- Missing values are imputed using statistics computed only from the training set.
- Descriptive and inferential calculations use available observations and report their actual sample sizes; missing target values are not imputed.
- Potential outliers are flagged by the 1.5-IQR rule in the numeric summary but are not automatically deleted. Plausible high-end GPUs are real observations; price uses a log transform, boxplots use robust quartiles, and manufacturer inference uses ranks.
- Kruskal-Wallis is used for manufacturer comparisons because it does not require normally distributed outcomes. It tests distributions/ranks and does not by itself identify which pair of manufacturers differs.
- Pearson tests measure linear association only; Bonferroni-adjusted p-values control family-wise error across each target's candidate features.
- The best model is selected using 5-fold cross-validation on the training set, then evaluated once on the 20% test set.
- The value `$14,999` is preserved as a valid data point; the price model uses a `log1p` transformation and converts predictions back to USD before computing prediction errors.
- The release data only extends to 2017, so the results are not representative of current GPU prices.
