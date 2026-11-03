# Text–election fusion feasibility

## Decision

**No valid text-to-constituency-year link currently exists in this repository.**

The available SemEval stance data is US English social-media text organized
around SemEval targets. It contains neither Indian parliamentary constituency
identifiers nor a defensible mapping to Lok Sabha election years. Geographic
or temporal links cannot be inferred from its topics, language, usernames, or
text. It must therefore not be joined to Indian constituencies, directly or
through fuzzy matching.

The repository also contains no user-supplied text feature file with explicit
`constituency_id` and `election_year` keys. Consequently, there is currently no
empirical basis for testing whether text improves real constituency election
prediction.

## Required evidence for a real fusion analysis

A user-supplied input must document:

- the provenance and collection period of the text;
- an explicit, auditable constituency assignment—not an inferred SemEval link;
- a mandatory `collection_end_year` (and preferably a precise timestamp)
  allowing features to be restricted
  to information available before the predicted election;
- aggregation rules, coverage, missingness, and protections against duplicated
  or cross-constituency content; and
- columns keyed uniquely by `(constituency_id, election_year)` containing only
  aggregate sentiment/stance values and embeddings, not raw text.

The fusion loader rejects duplicate keys and rows dated after the associated
prediction cutoff. Any raw source text must remain under `outputs/local/` and
must not appear in reports or MLflow parameters.

## Current evaluation branch

Because the required link is absent, `scripts/run_fusion.py` runs a seeded,
clearly labeled synthetic benchmark. It injects a known text-correlated signal
and verifies the tabular-only, late-fusion, and PyTorch early-fusion machinery
under expanding-window validation.

**Synthetic results validate the pipeline and ablation calculations only. They
are not evidence that text improves real election prediction, that the chosen
features exist in practice, or that any gain would transfer to Indian
constituencies.**
