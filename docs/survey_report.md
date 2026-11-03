# Survey analysis report

**Synthetic methodology demonstration.**

This report uses generated synthetic respondents and is not evidence about any real population or election.

## Weighted estimate

Weighted vote-intention proportion: `0.5464` (design-effect-aware 95% CI `0.5008`–`0.5913`).
Kish effective sample size: `461.15`; design effect: `1.301`.

## Weighted cross-tab

Design-effect-adjusted chi-square: `12.6232`, p-value: `0.0004`.
This is a documented Kish-adjusted Pearson approximation, not a full Rao–Scott correction because PSU and stratum identifiers are not in the minimal schema.

## Raking

Converged: `True` after `2` iterations; maximum relative margin error `0.00000003`.

## Weighted logistic regression

| Term | Coefficient | Standard error |
|---|---:|---:|
| const | -0.0620 | 0.1424 |
| age_group_35+ | 0.6722 | 0.1665 |
| gender_M | -0.1646 | 0.1665 |

## Simplified MRP

The demonstration uses statsmodels penalized binomial GLM with demographic indicators and L2-shrunk constituency effects as a pragmatic partial-pooling approximation, followed by poststratification. It is not a fully Bayesian multilevel model.

| Constituency | MRP estimate | Known synthetic truth | Absolute error |
|---|---:|---:|---:|
| C1 | 0.4652 | 0.4583 | 0.0069 |
| C2 | 0.5803 | 0.5886 | 0.0083 |
| C3 | 0.5903 | 0.6667 | 0.0764 |
