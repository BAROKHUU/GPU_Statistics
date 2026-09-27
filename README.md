# GPU Data Analysis Project: Probability and Statistics

The input data comes from `data/All_GPUs.csv`.
The program analyzes the dataset and predicts `Max_Power` (W) and
`Release_Price` (USD).

## Project structure

```text
code_Statistics/
├── data/
│   └── All_GPUs.csv           # Raw dataset
├── src/
│   ├── config.py              # Input variables and two tasks
│   ├── cleaning.py            # Read CSV, normalize units, handle unreleased prices
│   ├── statistics.py          # Descriptive statistics and correlation analysis
│   ├── visualization.py       # Plot generation
│   ├── regression.py          # Preprocessing, Ridge, and Random Forest models
│   └── analysis.py            # Train/test split, evaluation, and result export
├── results/                   # Generated outputs after running the project
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
`python main.py --data /path/to/All_GPUs.csv --output /path/to/output`.

The program writes `.csv`, `.png`, and `summary.txt` files to `results/`.
Missing values are imputed using statistics learned only from the training set.
The best model is selected using 5-fold cross-validation on the training set,
then evaluated once on the 20% test set.

Among 556 GPUs with original prices, 11 GPUs labeled `[not released]` are excluded from the price prediction task.
The value $14,999 is preserved; the price model uses a `log1p` transformation and converts predictions back to USD before computing errors.
The release data only extends to 2017, so the results are not representative of current GPU prices.
