#!/usr/bin/env python
# coding: utf-8

from confdl import reformat_json, Vocabulary

import matplotlib as mpl
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.axis"] = "x"   # only vertical lines
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.alpha"] = 0.3
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd

import sklearn 
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import json
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy.special as jsp 
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS

DEVICE_COUNT = 1
numpyro.set_host_device_count(DEVICE_COUNT)


class Dataset:

    def __init__(self, j_data, rng=np.random.RandomState(23)):
        self.rng = rng
        self.Is = []
        self.Rs = []
        self.Ys = []
        self.items = Vocabulary()
        self.raters = Vocabulary()

        for i, j_rows in j_data.items():
            for rec in j_rows:
                iid = self.items(rec['id'])
                rid = self.raters(rec['rater'])
                y = rec['label']
                self.Is.append(iid)
                self.Rs.append(rid)
                self.Ys.append(y) 

    def get_mask(self, dropout_rates=None):
        mask = np.ones_like(self.Y)
        if dropout_rates is None:
            return mask
        for rater, drop_rate in dropout_rates.items():
            rid = self.raters(rater)
            coord = self.R == rid
            values = self.rng.uniform(0, 1, size=mask[coord].size) > drop_rate
            mask[coord] = values
        return mask

    @property
    def I(self):
        """
        The sequence of observed item ids
        shape [num_observations]
        """
        return np.array(self.Is)

    @property
    def R(self):
        """
        The sequence of observed rater ids
        shape [num_observations]
        """
        return np.array(self.Rs)

    @property
    def Y(self):
        """
        The sequence of observed labels
        shape [num_observations]
        """
        return np.array(self.Ys)

    @property
    def num_items(self):
        return len(self.items)

    @property
    def num_raters(self):
        return len(self.raters)

    @property
    def num_observations(self):
        return len(self.Y)

    def get_counts(self, raters=[]):
        rids = set()
        for rater in raters:
            if isinstance(rater, str):
                rid = self.raters(rater)
            else:
                rid = rater
            rids.add(rid)

        O0 = np.zeros(self.num_items)
        O1 = np.zeros(self.num_items)
        for i, r, y in zip(self.I, self.R, self.Y):
            if r in rids:
                if y:
                    O1[i] += 1
                else:
                    O0[i] += 1
        return O1, O0


def mixture_model(rater_idx, item_idx, oracle, num_items, num_raters, y_obs=None, mask=None, O1=0, O0=0, idealized=None, 
                  item_prior=[1., 1.], oracle_prior=[0.5, 0.5], nonoracle_prior=[1., 1.]):
    """
    For readability, 
        M = num_observations
        N = num_items
        R = num_raters


    y_obs: (M,) observed ratings
    mask: (M,) 1 if observed
    rater_idx: (M,)
    item_idx: (M,)
    num_items: number of unique items
    raters: R strings of the kind O/oracle-alias, A/annotator-alias, Q/quantifier-alias
    O1: (N,) number of times each item was labelled True by a reliable source; use if available
    O0: (N,) number of times each item was labelled False by a reliable source; use if available    
    item_prior: larger values (eg, [5., 5.] or [10., 10.]) will make the analysis more 'eagerly confident'
    oracle_prior: a sparse Beta prior is appropriate (eg, [0.5, 0.5]), since oracle annotators are usually reliable
    nonoracle_prior: a flat Beta prior is appropriate (eg, [1., 1.])
    """

    # Item prob's prior
    # []
    alpha_theta = numpyro.sample("alpha_theta", dist.Exponential(item_prior[0]))
    # []
    beta_theta  = numpyro.sample("beta_theta",  dist.Exponential(item_prior[1]))
    # alpha_theta, beta_theta = 0.1, 0.1

    if idealized is None:
        idealized = jnp.zeros_like(oracle)

    with numpyro.plate("items", num_items):
        # Probability that z_i is True
        # [N]
        theta = numpyro.sample("theta", dist.Beta(alpha_theta + O1, beta_theta + O0))


    with numpyro.plate("raters", num_raters):   
        alpha_p1 = jnp.where(oracle, oracle_prior[0], nonoracle_prior[0])
        beta_p1 = jnp.where(oracle, oracle_prior[0], nonoracle_prior[0])

        alpha_p0 = jnp.where(oracle, oracle_prior[0], nonoracle_prior[0])
        beta_p0 = jnp.where(oracle, oracle_prior[0], nonoracle_prior[0])

        # P(y=1|z=1) "sensitivity"
        p1 = numpyro.sample("p1_", dist.Beta(alpha_p1, beta_p1))                

        # P(y=0|z=0) "specificity"
        p0 = numpyro.sample("p0_", dist.Beta(alpha_p0, beta_p0))


        p1 = jnp.where(jnp.logical_and(oracle, idealized), jnp.ones_like(p1), p1)            
        p0 = jnp.where(jnp.logical_and(oracle, idealized), jnp.ones_like(p0), p0)

        numpyro.deterministic("p1", p1)
        numpyro.deterministic("p0", p0)

    # Observation model
    theta_i = theta[item_idx]
    p1_r = p1[rater_idx]
    p0_r = p0[rater_idx]    
    # marginal prob of y[r,i] = 1
    phi_ir = numpyro.deterministic("phi_ir", theta_i * p1_r + (1 - theta_i)*(1 - p0_r))

    if mask is None:
        mask = jnp.ones(item_idx.size)
    numpyro.sample(
        "obs",
        dist.Bernoulli(phi_ir).mask(mask),
        obs=y_obs
    )

def posterior_inferences(samples):
    
    inferences = dict()

    # theta needs to go from [S, N] to [S, 1, N]
    # p  needs to go from [S, R] to [S, R, 1]
    # so their product has shape [S, R, N]

    # [S, R, N]
    # P(z=1,y=1)
    TP = samples['theta'][:,None,:] * samples['p1'][:,:,None]
    # [S, R, N]
    # P(z=0,y=)
    TN = (1-samples['theta'][:,None,:])*samples['p0'][:,:,None]

    # [S, R, N]
    # P(z=1,y=0)
    FN = samples['theta'][:,None,:] * (1-samples['p1'][:,:,None])
    # [S, R, N]
    # P(z=0,y=1)
    FP = (1-samples['theta'][:,None,:]) * (1-samples['p0'][:,:,None])

    # [S, R]
    inferences['p0'] = samples['p0']
    inferences['p1'] = samples['p1']
    inferences['TP'] = jnp.mean(TP, -1)
    inferences['TN'] = jnp.mean(TN, -1)
    inferences['FN'] = jnp.mean(FN, -1)
    inferences['FP'] = jnp.mean(FP, -1)
    inferences['acc'] = jnp.mean(TP + TN, -1)
    inferences['err'] = jnp.mean(FP + FN, -1)

    # These two inferences are "approximate" for they are not linear functions of the inferred parameters (thetas, p0, p1, etc)
    # rather they are nonlinear functions of observable _data_
    # to estimate them from observable data, we would have to simulate observations from the inferred latents
    # (eg, using the Predictive object above)
    inferences['F1'] = jnp.mean(2*TP / (2*TP + FP + FN), -1)
    inferences['MCC'] = jnp.mean((TP*TN - FP*FN)/(jnp.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))), -1)

    return inferences

def sort_by_median(idata, variable, label, showing):
    """
    idata: from arviz
    variable: such as acc or F1
    label: such as rater, judge, system
    showing: list of label values to show
    """
    med = idata.posterior[variable].median(dim=("chain", "draw"))
    med_sel = med.sel({label: showing})
    med_sorted = med_sel.sortby(med_sel)        # ascending by median
    sorted_showing = med_sorted[label].values
    return list(sorted_showing)

def run_experiment(
    basefile,    
    output_folder="inferences/",
    experiment_suffix="",
    title_prefix="",
    item_prior=[0.5, 0.5],
    oracle_prior=[1., 0.5],
    nonoracle_prior=[1., 1.],
    mcmc_params={
        "num_warmup": 1000,
        "num_samples": 2000,
        "num_chains": 2,
    },
    az_stats_params={
        "point_estimate": "median",
        "ci_kind": 'hdi',
        "ci_probs": (0.5, 0.95),
    },
    rename_judges={
        'llm_as_judge_gpt-5.4-mini': 'gpt', 
        'llm_as_judge_llama-8b-instruct': 'llama', 
        'embedding_similarity': 'esim',
        'token_f1': 'tokf',
        'bert_score': 'bert',
        'human_judge1': 'human1',
        'human_judge2': 'human2',
        'exact_match': 'EM',
        'rouge_l': 'rouge',
    },
    np_seed=42,
    prng_seed=0
):
    
    input_file = f"data/{basefile}.json"
    experiment = f"{basefile}{experiment_suffix}"

    j_rows, s_rows = reformat_json(
        input_file,
        # shortening some names
        rename=rename_judges,
        # analysing a subset of the data
        skip_all_quantifiers=True
    )
    
    data = Dataset(j_rows, rng=np.random.RandomState(np_seed))
    
    print(f"Analysing {data.num_items} items and {data.num_raters} raters")
    for r, rater in enumerate(data.raters):
        print(f"\t{r}/{rater}")
    print(f"items={data.I.shape} raters={data.R.shape} labels={data.Y.shape}")
    
    
    # chosen_judges = ['bleu/0.6', 'EM', 
    #         'tokf/0.4', 'rouge/0.5',
    #         'tokf/0.2', 'rouge/0.3',
    #         'bert/0.7','esim/0.5', 'esim/0.3',
    #         'esim/0.4', 'rouge/0.1', 'llama', 'gpt']
    # chosen_oracles = ['human1'] 
    # if 'human2' in data.raters.obj2int:
    #     chosen_oracles += ['human2']
    # chosen_judges + chosen_oracles
        
    kernel = NUTS(mixture_model)
    
    
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_params['num_warmup'],
        num_samples=mcmc_params['num_samples'],
        num_chains=mcmc_params['num_chains']
    )
    rng_key = random.PRNGKey(prng_seed)
    
    # Sparse Beta prior on theta
    
    # The Beta distribution on thetas is the posterior of a shallow Beta-Bernoulli model
    # the sufficient statistics are the human counts for True/False for each item
    # If you disable this (use 0 counts) then you will need to find other ways to break the 
    # symmetry between oracle and non-oracle groups
    O1, O0 = data.get_counts(['human1', 'human2'])
    
    
    # This tells the BDA model which judges draw (p0,p1) from which priors
    oracle_flags = np.array([rater.startswith('human') for rater in data.raters])
    # This tells the BDA model whether any judge can be assumed to be 'idealized' (p0=p1=1)
    idealized_flags = np.array([False for rater in data.raters])
    
    mcmc.run(
        rng_key,    
        rater_idx=data.R,
        item_idx=data.I,        
        oracle=oracle_flags,
        num_items=data.num_items,
        num_raters=data.num_raters,    
        y_obs=data.Y,  # use None for predictions and data.Y for posterior inference
        mask=None, # use None for predictions and data.get_mask() for posterior inference
        #you can drop observations like this (for example, to test the effect of data size)
        #mask=data.get_mask({'human_judge1': 0, 'human_judge2': 0}),    
        idealized=idealized_flags,
        # use these to play with item-level posteriors as priors
        O1=O1,
        O0=O0,
        item_prior=item_prior,
        oracle_prior=oracle_prior,
        nonoracle_prior=nonoracle_prior,
    )
        
    mcmc.print_summary()
    
    posterior_data = az.from_numpyro(
        mcmc,
        dims={
            'alpha_theta': [],
            'beta_theta': [],
            'p0_': ['judge'],
            'p0': ['judge'],
            'p1_': ['judge'],
            'p1': ['judge'],
            'theta': ['item'],
        },
        coords={
            "judge": data.raters, 
            "item": data.items
        }   
    )
    
    # az.plot_forest(
    #     posterior_data, 
    #     var_names=['alpha_theta', 'beta_theta'],
    #     aes={"color": ["__variable__"]},
    #     aes_by_visuals={"labels": ["color"]},
    #     **az_stats_params
    # )
    
    # az.plot_forest(
    #     posterior_data, 
    #     var_names=['p0', 'p1'],
    #     shade_label="judge",
    #     aes={"color": ["__variable__"]},
    #     aes_by_visuals={"labels": ["color"]},
    #     **az_stats_params
    # )
    
    posterior_samples = mcmc.get_samples(group_by_chain=False)
    # print("posterior samples shapes:")
    # for k, v in posterior_samples.items():
    #     print(k, v.shape)
        
    predictive = Predictive(mixture_model, posterior_samples)
    pred_samples = predictive(
        random.PRNGKey(prng_seed + 1),     
        rater_idx=data.R,
        item_idx=data.I,    
        oracle=oracle_flags,
        num_items=data.num_items,
        num_raters=data.num_raters,
        y_obs=None,
        mask=None,
        idealized=idealized_flags,
        O1=O1, 
        O0=O0,
        item_prior=item_prior,
        oracle_prior=oracle_prior,
        nonoracle_prior=nonoracle_prior,
    )
    
    # for k, v in pred_samples.items():
    #     print(k, v.shape)
    
    
    pred_dict = {'acc': []}
    for rid, rname in enumerate(data.raters):
        predaccs = (pred_samples['obs'][:,data.R==rid] == data.Y[data.R==rid]).mean(-1)
        pred_dict['acc'].append(predaccs)    

    
    pred_data = az.from_dict(
        {
            'posterior': {'acc': np.stack(pred_dict['acc'], -1).reshape(mcmc_params['num_chains'], mcmc_params['num_samples'], -1)}
        },    
        dims={
            'acc': ['judge'],
        },
        coords={
            "judge": data.raters,
        },    
    )
    _ = az.plot_forest(
        pred_data,
        var_names=['acc'],
        combined=True,
        shade_label='judge',
        **az_stats_params
    )  
    plt.title(f"{title_prefix}Predictive Accuracy")
    plt.savefig(f"{output_folder}/{experiment}.predictive-acc.pdf")
    
    # _ = az.plot_forest(
    #     pred_data.sel(judge=chosen_judges + chosen_oracles),    
    #     var_names=['acc'],
    #     combined=True,
    #     shade_label='judge',
    #     **az_stats_params
    # )
    # # _ = plt.title("Posterior Predictive")
    # # _ = plt.tight_layout()
    # plt.savefig(f'inferences/{experiment}.predictive-acc-chosen.pdf')
    
    
    # # Analysis    
    inf = posterior_inferences(posterior_samples)    
    inf_data = az.from_dict(
        {
            'posterior': {k: v.reshape(mcmc_params['num_chains'], mcmc_params['num_samples'], -1) for k, v in inf.items()}
        },
        dims={
            'p0': ['judge'],
            'p1': ['judge'],
            'TP': ['judge'],
            'TN': ['judge'],
            'FN': ['judge'],
            'FP': ['judge'],
            'acc': ['judge'],
            'err': ['judge'],
            'F1': ['judge'],
            'MCC': ['judge'],
        },
        coords={
            "judge": data.raters,
        },    
    )

    az.plot_forest(
        inf_data,
        coords={"judge": sort_by_median(inf_data, 'F1', 'judge', list(data.raters))},
        var_names=['F1'],
        shade_label="judge",
        combined=True,
        **az_stats_params
    )
    plt.title(f"{title_prefix}Posterior F1")
    _ = plt.savefig(f"{output_folder}/{experiment}.posterior-F1.pdf")
    
    # az.plot_forest(
    #     # inf_data.sel(judge=chosen_judges + chosen_oracles),
    #     inf_data,
    #     coords={"judge": sort_by_median(inf_data, 'F1', 'judge', chosen_judges + chosen_oracles)},
    #     var_names=['F1'],
    #     shade_label="judge",
    #     combined=True,    
    #     # figure_kwargs={"width_ratios": [0.4, 1], "layout": "none"},
    #     figure_kwargs={"width_ratios": [1, 2], "figsize": (6, 4)},
    #     **az_stats_params
    # )
    # # _ = plt.title("Posterior F1")
    # # _ = plt.tight_layout()
    # _ = plt.savefig(f"inferences/{experiment}.posterior-F1-chosen.pdf")
    

    az.plot_forest(
        inf_data,
        var_names=['TP', 'TN'],    
        # figure_kwargs={"width_ratios": [1, 2], "layout": "none"},
        aes={"color": ["__variable__"]},
        aes_by_visuals={"labels": ["color"]},
        # shade_label="judge",
        combined=True,
        **az_stats_params
    )
    plt.title(f"{title_prefix}Predictive TPR and TNR")
    plt.savefig(f"{output_folder}/{experiment}.posterior-TPRTNR.pdf")
    
    az.plot_forest(
        inf_data,
        var_names=['FP', 'FN'],    
        # figure_kwargs={"width_ratios": [1, 2], "layout": "none"},
        aes={"color": ["__variable__"]},
        aes_by_visuals={"labels": ["color"]},
        # shade_label="judge",
        combined=True,
        **az_stats_params
    )
    plt.title(f"{title_prefix}Predictive FPR and FNR")
    plt.savefig(f"{output_folder}/{experiment}.posterior-FPRFNR.pdf")
    

if __name__ == '__main__':

    datasets = [
        'trivia_qa',
        'ambig_qa_multiple_qas',
        'ambig_qa_single_answer'
    ]
    generators = [
        'llama-8b-instruct',
        'qwen-7b-instruct',
        'qwen-0.5b-instruct'
    ]
    # datasets = [
    #     'trivia_qa_llama-8b-instruct',
    #     'trivia_qa_qwen-7b-instruct',
    #     'trivia_qa_qwen-0.5b-instruct',
    #     'ambig_qa_multiple_qas_llama-3b-instruct',
    #     'ambig_qa_multiple_qas_llama-8b-instruct',
    #     'ambig_qa_multiple_qas_qwen-0.5b-instruct',
    #     'ambig_qa_multiple_qas_qwen-7b-instruct',
    #     'ambig_qa_single_answer_llama-3b-instruct',
    #     'ambig_qa_single_answer_llama-8b-instruct',
    #     'ambig_qa_single_answer_qwen-0.5b-instruct',
    #     'ambig_qa_single_answer_qwen-7b-instruct'
    # ]

    for dataset in datasets:
        for generator in generators:
            name = f"{dataset}_{generator}"
            print("BASE FILE:", name)
            run_experiment(
                name,
                output_folder="inlg2026",
                title_prefix=f"dataset={dataset} generator={generator} | ",
            )
            break
        break