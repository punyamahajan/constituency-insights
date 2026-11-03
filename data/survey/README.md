# User-provided survey data

No real survey is downloaded or bundled. Place a user-provided CSV outside
version control (recommended: `data/raw/survey/survey.csv`) and point a YAML
configuration at it.

Required schema:

| Field | Requirement |
|---|---|
| Respondent ID | Non-empty and unique |
| Demographics | One or more configured columns such as age group, gender, region, and constituency |
| Weight | Numeric, finite, and strictly positive |
| Stance or vote intention | Non-missing; binary for logistic models |

Column names are configured under `schema` rather than assumed. Population
margins used for raking must come from an explicitly supplied source and must
cover every sampled category. Poststratification data must contain the same
demographic cells, a constituency column, and a population-count column.

`data/demo/survey_demo.csv` is generated synthetic data for methodology
demonstration only. It is not evidence about any real electorate.
