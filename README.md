# Algorithm agnostic evaluation for (disease) outbreak detection

## Motivation

In the field of disease outbreak detection, qualitatively different families of algorithms are used such as Farrington Flexible and SatScan. However, comparing the performance of different algorithm families is not trivial. Inputs and outputs differ vastly differ between them and therefore make a clear quantitative comparison difficult.

Our score offers a solution to make formerly non comparable approaches comparable.

## Installation

To run the notebooks and to be able to use the `Scorer` class, you need to install the packages listed in `env.yml`. If you use conda, simply run

```
conda env create -f environment.yml
```

Afterwards, run

```
conda activate scoring
```

to activate the conda environment.

## How to use the scorer

### What do I need?

The scorer requires the following inputs:

#### data

This is a `pandas.DataFrame` that contains case counts of a certain infectious disease where each row aggregates the case numbers along the lowest possible resolution over some dimensions.

Such a table could look like this:

| x_week | x_county | one | two | endemic | non_cases |
| ------ | -------- | --- | --- | ------- | --------- |
| 0      | 0        | 0   | 0   | 0       | NaN       |
| 0      | 1        | 0   | 1   | 0       | NaN       |
| 1      | 0        | 1   | 1   | 1       | NaN       |
| 1      | 1        | 0   | 2   | 1       | NaN       |

where the dimension columns are prefixed with `x_` and the outbreak labels, i.e., the cases that are caused by one underlying event don't. We shall refer to `x_` columns as coordinate columns.

In this table `one` and `two` are the IDs for an outbreak and `endemic` are cases that don't belong to an outbreak and `non_cases` contain a 1 if a cell has no cases and 0 otherwise. The `non_cases` and `endemic` column are mandatory in for this table.

#### signals

This is also a `pandas.DataFrame` that should have the same length and thus dimensions as `data` but where each non-coordinate column is the significance $w$ of an detected outbreak per cell:

| x_week | x_county | w_A | w_B |
| ------ | -------- | --- | --- |
| 0      | 0        | 0   | 0   |
| 0      | 1        | 0   | 1   |
| 1      | 0        | 1   | 1   |
| 1      | 1        | 0   | 2   |

Signal columns are indicated with a prefixed `w_`.

#### data_labels

To tell the `Scorer` which columns contain the relevant outbreak information, we need to specify the `data_labels`. In the `data` table in the above example the respective `data_labels` would be `one` and `two`. Thus, the required input would be

```python
data_labels=["one", "two"]
```

#### missing_signal_agg

Finally, we might need to generate two missing signals. In the example above, we see that we only the columns `w_A` and `w_B` but not `w_endemic` and `w_non_case`. There are many such algorithms that don't specifically generate a signal for these `data_labels`. In this case, we need to estimate these signals.

This is done by assuming that if an algorithm has a certain signal $w_{i,j}$ for cell $i,j$ and there are no cases is this cell, `w_non_case` is $1-w_{i,j}$. If this cell contains cases, then `w_endemic` is $1-w_{i,j}$.

If we have several signals, we also generate many signals for `w_endemic` and `w_non_case`. In our example, we have two signals, therefore we would generate signals named `_A_endemic`, `_B_endemic`, `_A_non_case`, and `_B_non_case`. How do we aggregate these signals?

The `missing_signal_agg` allows to set the aggregation strategy. The default is "min" which would take the lowest signal for `non_case` and `endemic` respectively from all generated signals per cell.

### The Scorer

Finally, if you have the `data`, `signals`, `data_labels`, and `missing_signal_agg`, using the scorer comes down to runnning:

```python
s = Score(
    df_cases,
    signals_without_endemic_non_case,
    ["one", "two", "three"],
    missing_signal_agg="min",
)
# Shows performance per data_label as confusion matrices as tables and plots.
s.class_based_conf_mat()

# Aggregates the performances of all data_labels to one single score
s.mean_score()

# Shows underlying scores per cell.
s.eval_df()
```
