Here is a description of the analytical metrics we report in our paper, and how we generate them from the computed raw statistics in the CSV files in this directory.

The tables we provide here were generated using the `--extended-metrics` flag available in [VLHub](https://github.com/penfever/vlhub/) and raw statistics computed over the datasets themselves.

For any given model we wish to analyze, we consider the following metrics:

| Metric Name | Fields in Table | Semantic Name | Computed using | 
| -------- | ------------- | ------------ | ---------- |
| **Coverage Classes** | 'Class Count' | How many classes in the label set are present in the dataset | Stats |
| **LT@500** | 'Class Count' | How many classes in the dataset contain <500 samples | Stats |
| **LS@5pct** | 'Class Count' | What percentage of dataset samples the top 5% of classes contain | Stats |
| **ClipScore** | None | ClipScore | Images, quality-metrics.py |
| **ClipIQA** | None | ClipIQA | Images, quality-metrics.py |
| **Inception** | None | Inception Score | Images, quality-metrics.py |
| **CMMD** | None | CMMD | Images, quality-metrics.py |
| **R:P, CC** | 'Precision', 'Class Count' | Correlation | Stats |
| **R:A, CS** | 'accuracy', 'skewness of normalized confusion scores' | Correlation | Stats |
| **R:INA, A** | 'accuracy (in1000)', 'accuracy' | Correlation | Stats |
| **R:P, R** | 'Precision', 'Recall' | Correlation | Stats |
| **R:INAV, AV** | 'Class Count (in1000)', 'Class Count' | Correlation | Stats |