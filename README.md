# Dependencies

Use python3.14

```bash
pip install jax numpyro arviz seaborn pandas scikit-learn
```

Download [rbo.py](https://github.com/dlukes/rbo)

# Guide

## NLG-vs-UQ-eval.ipynb

Plot F1 vs. Rank Correlation (or human-judge-powered AUROC rankings and auto-judge-powered AUROC rankings) as a function of Judge.

See plots under `otherstuff/`

## BDA-F1.ipynb

Load human judgements and automated judgements for a dataset-generator pair and analyse them in a Bayesian model whose latent parameters capture TPR, TNR, FPR and FNR, and hence can be used to infer posterior accuracy and posterior of F1 of each automated judge relative to the human data. 

See plots under `inferences/`

To generate all figures for all datasets, you can use

```bash
python pipeline-bda-f1.py 2>&1 | tee inlg2026/log.txt
```

These figures will contain all judges, which is too much for a paper. To generate versions that filter from the available judges, use the notebook.
