# Outcome model card

## Intended use

The API accepts constituency-level election features and returns a predicted
vote share with a split-conformal interval. It is intended for research
prototyping and decision support with human oversight.

## Training data and metrics

The outcome model is designed for locally supplied constituency election and
demographic data. No real outcome metrics are reported because the trained artifact is unavailable.

## Limitations

Conformal coverage depends on the calibration exchangeability assumptions and
can fail under structural political or demographic shifts. The current
repository does not contain a real trained outcome artifact, so the endpoint
returns HTTP 503 rather than fabricate predictions.

## Ethical considerations

Election forecasts can affect political strategy, public trust, and resource
allocation. Do not present intervals as certainties or use predictions to
target protected groups. Validate provenance, fairness, and uncertainty before
deployment.
