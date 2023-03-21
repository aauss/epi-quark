![Buildstatus](https://github.com/aauss/epi-quark/actions/workflows/ci.yml/badge.svg?branch=master)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://codecov.io/gh/aauss/epi-quark">
  <img src="https://codecov.io/gh/aauss/epi-quark/branch/master/graph/badge.svg?token=U7VTC00G71"/>
</a>
![CodeQL](https://github.com/aauss/epi-quark/workflows/CodeQL/badge.svg)
<a href="https://opensource.org/licenses/MIT"><img alt="Code style: black" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

# Algorithm agnostic evaluation for (disease) outbreak detection

## Motivation

In the field of disease outbreak detection, different types of algorithms are frequently used such as Farrington or SaTScan. Measuring and comparing the performances of different algorithm families is made difficult by a lack of a common approach in the community, different types of output, as well as different settings of implementation.

The framework implemented in epi-quark to compute scores allows such evaluation and comparisons.
It is based on an heuristic to compare algorithm outputs at different aggregation scales and on expert-annotated infection cases.
Further details can be found in an [accompanying paper](https://doi.org/10.1101/2022.03.16.22272469).

## Installation

To use epi-quark and compute scores, run the notebooks, build the docs, run the tests, you need to install the packages listed in `env-dev.yml`. If you use conda, run

```
conda env create -f env-dev.yml
```

Afterwards, run

```
conda activate epi-quark
```

to activate the conda environment. You should be ready to go.

### Docs

To build the documentation, run

```
cd docs/
make html
```

which will build the sphinx-based HTML documentation. Open the index page `docs/build/html/index.html` to access and navigate it.

### Pre-commit

If you want to check your code before committing it, you can use `pre-commit` to run CI checks set up for this repo.

To install git hook scripts to evaluate your code, run

```
pre-commit install
```

This might take a moment. If the setup was successful, your code is checked before you can commit it. You can read more about it [here](https://pre-commit.com/).

## How to use the scorer

### What do I need?

The scorer requires the following inputs:

#### **data**

This is a `pandas.DataFrame` that contains case counts of some infectious disease where each row aggregates the case numbers along the lowest possible resolution over some dimensions.

Such a table could look like this:

| week | county | data_label | value |
| ---- | ------ | ---------- | ----- |
| 0    | 0      | endemic    | 0     |
| 0    | 1      | endemic    | 3     |
| 0    | 0      | one        | 0     |
| 0    | 1      | one        | 1     |
| 0    | 0      | two        | 1     |
| 0    | 1      | two        | 1     |

where the data labels are extracted from the `data_label` column, its values form the `value` column, and the remaining columns are treated as coordinates.

In this table, `one` and `two` are the labels for an outbreak and `endemic` are cases that don't belong to an outbreak. You data needs to at least contain data with the data_label `endemic`. On the other hand, `non_cases` only occur if all other labels have a value of `0`. This is, however, handled internally by the package and therefore no `data_label` should be named `non_cases`.

The coordinate system of the data DataFrame is considered complete, i.e., you should make sure that all relevant cells are contained in the coordinate columns. Also,  each cell should contain one entry for each data_label that you want to consider for your analysis.

#### **signals**

This is also a `pandas.DataFrame` and its coordinate system needs to be a subset of the coordinates defined the `cases` DataFrame, i.e. it may contain other dimensions or more coordinate points which, however, are ignored for the analysis. Instead of a `data_label` column, this DataFrame should contain a `signal_label` column which contains the different signal labels generated by the outbreak detection algorithms. The significance of the signal is entered in to the `value` column and should be normalized to a value between 0 and 1. Please make sure to always include `endemic` and `non_case` signals.

| week | county | signal_label | value |
| ---- | ------ | ------------ | ----- |
| 0    | 0      | endemic      | 0     |
| 0    | 1      | endemic      | 0.2   |
| 0    | 0      | non_case     | 0.5   |
| 0    | 1      | non_case     | 0     |
| 0    | 0      | A            | 0     |
| 0    | 1      | A            | 1     |
| 0    | 0      | B            | 0.5   |
| 0    | 1      | B            | 1     |

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
# 0  0.0  0.0          A    0.0
# 1  0.0  1.0          A    0.0
# 2  0.0  2.0          A    0.0
# 3  0.0  3.0          A    0.0
# 4  0.0  4.0          A    0.0

score(cases, signals, "r2")
# {'endemic': -0.013702460850111953,
#  'non_case': 0.7996794871794872,
#  'one': 0.3055555555555556,
#  'three': -0.7795138888888888,
#  'two': -0.17753623188405832}

# Some metrics require binary values such as F1. In this case, set thresholds.
thresholded_metric = score(cases, signals, threshold_true=0.5, threshold_pred=0.5, metric="f1")

# If you want to weight cells with more cases higher than others, use the `weighting` parameter.
case_weighted = score(cases, signals, "r2", weighting="cases")

# You can also weight by spatio-temporal accuracy of the detected outbreak. Just assign which column is time and
# which is the spacial weighting dimension.
timespace_weighted = timespace_weighted = score(
        cases,
        signals,
        "r2",
        weighting="timespace",
        time_space_weighting={"x1": 1, "x2": 1.5},
        time_axis="x1",
    )
```
