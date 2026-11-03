# Unsupervised and Semi-Supervised Text Track

Generated exclusively from MLflow runs over the official SemEval training split and its fixed derived validation partition. The official test split was not accessed.

## Topic-model sweep

Topic quality is compared quantitatively using NPMI coherence; topic words were not selected or judged by eye.

| Method | Requested topics | Actual topics | NPMI coherence | MLflow run |
| --- | ---: | ---: | ---: | --- |
| BERTopic | 10 | 10 | -0.965483 | `31367c2acc1a42298c6e489b96ecda8f` |
| BERTopic | 15 | 15 | -0.977661 | `7b74974bc37b49e686f2ddbe80a475ea` |
| BERTopic | 20 | 20 | -0.973917 | `3aa71a2a19c34549bd7edcdcec23941a` |
| BERTopic | 5 | 5 | -0.948750 | `3dd40f7fbf324284bccf5ffe66733251` |
| LDA | 10 | 10 | -0.101660 | `1d7af7d2533645268507bbe051c3fd85` |
| LDA | 15 | 15 | -0.199022 | `1b42bc91ce1e44749672053190608750` |
| LDA | 20 | 20 | -0.202972 | `f02645c0e6a04c00887c9497507efa66` |
| LDA | 5 | 5 | -0.102297 | `84fa7449e4c24d468fe719a6fa5ee9f3` |

## Embedding clustering

Stability is mean pairwise adjusted Rand index across the configured ten seeded UMAP-and-clustering runs. Silhouette and Davies–Bouldin exclude HDBSCAN noise points.

| Method | Configuration | Clusters | Noise fraction | Silhouette | Davies-Bouldin | Stability ARI | MLflow run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| HDBSCAN | min_cluster_size=10 | 21 | 0.308854 | 0.473316 | 0.634687 | 0.684431 | `1e9950800bd2475c89dfc2e2ca8b31f1` |
| HDBSCAN | min_cluster_size=25 | 10 | 0.299245 | 0.604067 | 0.485057 | 0.797797 | `9bea5b38192e4ca6ade176d988c9c3ab` |
| HDBSCAN | min_cluster_size=50 | 6 | 0.201098 | 0.546029 | 0.619324 | 0.766533 | `f63a413e5ba94bf2ba2360debcd0d7bf` |
| KMeans | k=10 | 10 | 0.000000 | 0.444273 | 0.932039 | 0.945151 | `63b1fdcca8834ec48bd51ee82bdb8c0b` |
| KMeans | k=15 | 15 | 0.000000 | 0.355572 | 1.075356 | 0.816805 | `4f9989066b05418e894feea72e151ef5` |
| KMeans | k=20 | 20 | 0.000000 | 0.370731 | 0.976939 | 0.785054 | `5a52ed54a2744948837c3fbf5a20d75d` |
| KMeans | k=5 | 5 | 0.000000 | 0.493167 | 0.754400 | 0.840007 | `f9888c5adb954f609feeab22771454cd` |

## Label-efficiency and one-round self-training

Pseudo-labels use a confidence threshold of 0.50. Each summary is calculated from three seeded MLflow runs.

| Label fraction | Labeled-only macro-F1 | With pseudo-labels macro-F1 | Mean pseudo-label selection rate | Source runs |
| ---: | ---: | ---: | ---: | --- |
| 10% | 0.216300 ± 0.000000 | 0.216300 ± 0.000000 | 0.234509 | `1eea90ec34e2405cbeb861bfe2ea6d42`, `0112d05af7c04ca598567208e3df9290`, `d7e37ac72dea474891558e2657d66feb` |
| 25% | 0.358541 ± 0.003828 | 0.357815 ± 0.000000 | 0.765962 | `def1c98ad5384c72a3c7d988947f08d6`, `34df89ad1de14449a51ddd3b4153c9df`, `84ac900437d5478589198e98905baa06` |
| 50% | 0.375480 ± 0.030146 | 0.376967 ± 0.037043 | 0.700686 | `011019f311004972a5c2e72e907dc269`, `8b0c86aa2e9e494aa53d9c24430c33f3`, `6448fa13b8214a07beb5ebb4dbbb30dc` |

All reported values are direct MLflow metrics or deterministic mean/standard-deviation summaries of the linked MLflow runs.
