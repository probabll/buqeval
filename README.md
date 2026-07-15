# Dependencies

Use python3.14

```bash
pip install numpyro arviz seaborn pandas scikit-learn
```

Download [rbo.py](https://github.com/dlukes/rbo)

# Guide

## NLG-vs-UQ-eval.ipynb

Plot F1 vs. Rank Correlation (or human-judge-powered AUROC rankings and auto-judge-powered AUROC rankings) as a function of Judge.

See plots under `otherstuff/`

## BDA-F1.ipynb

Load human judgements and automated judgements for a dataset-generator pair and analyse them in a Bayesian model whose latent parameters capture TPR, TNR, FPR and FNR, and hence can be used to infer posterior accuracy and posterior of F1 of each automated judge relative to the human data. 

See plots under `inferences/`
