# Stance model card

## Intended use

The API serves a frozen target-conditioned DistilBERT classifier for stance
screening on English political text paired with a target. It is an inference
service component, not an autonomous decision-maker.

## Training data and metrics

The model was trained on SemEval-2016 Task 6A US English Twitter data. Metrics
below are pulled from the generated `docs/final_test_results.md` report:

- Accuracy: `0.6413`
- Macro-F1: `0.6245`
- Official FAVOR/AGAINST F1: `0.6420`

## Limitations

SemEval is English-only and represents US political discourse. The model is not
validated for Hindi, Tamil, code-mixed content, Indian politics, constituency
discourse, or distribution shifts. Probabilities are not guaranteed calibrated.

## Ethical considerations

Political stance predictions can misclassify, stereotype, or amplify annotation
bias. Do not use outputs to target individuals, determine eligibility, or make
high-impact decisions without human review and domain-specific validation.
Request text is not logged by this service.
