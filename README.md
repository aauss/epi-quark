<a href="https://app.travis-ci.com/aauss/epi-quark.svg?branch=master">
        <img src="https://img.shields.io/circleci/project/github/badges/shields/master" alt="build status"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://codecov.io/gh/aauss/epi-quark">
  <img src="https://codecov.io/gh/aauss/epi-quark/branch/master/graph/badge.svg?token=U7VTC00G71"/>
</a>

# Algorithm agnostic evaluation for (disease) outbreak detection

## Motivation

In the field of disease outbreak detection, qualitatively different families of algorithms are used such as Farrington Flexible and SaTScan. However, comparing the performance of different algorithm families is not trivial. Inputs and outputs differ vastly between them and therefore make a clear quantitative comparison difficult.

Our score offers a solution to make formerly non-comparable approaches comparable.

## Installation

To run the notebooks and to be able to use our score, you need to install the packages listed in `env.yml`. If you use conda, simply run

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

#### **data**

This is a `pandas.DataFrame` that contains case counts of a certain infectious disease where each row aggregates the case numbers along the lowest possible resolution over some dimensions.

Such a table could look like this:

| x_week | x_county | data_label | value |
| ------ | -------- | ---------- | ----- |
| 0      | 0        | endemic    | 0     |
| 0      | 1        | endemic    | 3     |
| 1      | 0        | one        | 0     |
| 1      | 1        | one        | 1     |
| 2      | 0        | two        | 1     |
| 2      | 1        | two        | 1     |

where the data labels are extracted from the `data_label` column, its values form the `value` column, and the remaining columns are treated as coordinates.

In this table `one` and `two` are the labels for an outbreak and `endemic` are cases that don't belong to an outbreak. `non_cases` contain a 1 if a cell has no cases and 0 otherwise. This is handled internally and therefore no `data_label` should be named `non_cases`.

The coordinate system of the data DataFrame is considered complete, i.e., all relevant cells for a analysis should be found in the coordinate columns.

#### **signals**

This is also a `pandas.DataFrame` that should be a subset of the coordinates defined the `cases` DataFrame. Instead of a `data_label` column, this DataFrame should contain a `signal_label` column which contains the different signal labels generated by the outbreak detection algorithms. The significance of the signal is entered in to the `value` column. All `signal_labels` must start with a "w\_":

| x_week | x_county | signal_label | value |
| ------ | -------- | ------------ | ----- |
| 0      | 0        | w_endemic    | 0     |
| 0      | 1        | w_endemic    | 3     |
| 1      | 0        | w_A          | 0     |
| 1      | 1        | w_A          | 1     |
| 2      | 0        | w_B          | 1     |
| 2      | 1        | w_B          | 1     |

### API

```python
import pandas as pd
from epiquark import conf_matrix, score, timeliness

cases = pd.read_csv("tests/data/paper_example/cases_long.csv")
signals = pd.read_csv("tests/data/paper_example/imputed_signals_long.csv")

cases.head()
#     x1   x2 data_label  value
# 0  0.0  0.0        one      0
# 1  0.0  1.0        one      0
# 2  0.0  2.0        one      0
# 3  0.0  3.0        one      0
# 4  0.0  4.0        one      0

signals.head()
#     x1   x2 signal_label  value
# 0  0.0  0.0          w_A    0.0
# 1  0.0  1.0          w_A    0.0
# 2  0.0  2.0          w_A    0.0
# 3  0.0  3.0          w_A    0.0
# 4  0.0  4.0          w_A    0.0

score(cases, signals, "r2")
# {'endemic': -0.013702460850111953,
#  'non_case': 0.7996794871794872,
#  'one': 0.3055555555555556,
#  'three': -0.7795138888888888,
#  'two': -0.17753623188405832}

# Some metrics require binary values such as F1. In this case, set thresholds.
thresholed_metric = score(cases, signals, threshold_true=0.5, threshold_pred=0.5, metric="f1")

# If you want to weight cells with more cases higher than others, use the `weights` parameter.
case_weighted = score(cases, signals, "r2", weights="cases")

# You can also weight by spatio-temporal accuracy of the detected outbreak. Just assign which column is the time and
# which is the other weighting dimension.
timespace_weighted = timespace_weighted = score(
    cases, signals, "r2", weights="timespace", gauss_dims="x2", time_axis="x1"
)
```
