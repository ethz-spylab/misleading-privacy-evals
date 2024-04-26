# Evaluations of Machine Learning Privacy Defenses are Misleading

*[Michael Aerni](https://www.michaelaerni.com/), [Jie Zhang](https://zj-jayzhang.github.io/), [Florian Tramèr](https://floriantramer.com/) | Spy Lab (ETH Zurich)*

Official repository for the paper **Evaluations of Machine Learning Privacy Defenses are Misleading**.


## Abstract

> Empirical defenses for machine learning privacy forgo
the provable guarantees of differential privacy in the hope of achieving higher utility
while resisting realistic adversaries.
We identify severe pitfalls in existing empirical privacy evaluations
(based on membership inference attacks) that result in misleading conclusions.
In particular, we show that prior evaluations fail to characterize
the privacy leakage of the most vulnerable samples,
use weak attacks,
and avoid comparisons with practical differential privacy baselines.
In 5 case studies of empirical privacy defenses,
we find that prior evaluations underestimate privacy leakage by an order of magnitude.
Under our stronger evaluation, none of the empirical defenses we study
are competitive with a properly tuned, high-utility DP-SGD baseline
(with vacuous provable guarantees).


## Getting started

Install and activate the conda environment in `environment.yml`:

```bash
micromamba env create -f environment.yml -n misleading-privacy-evals
micromamba activate misleading-privacy-evals
```

The experiments rely on environment variables for convenience.
To configure those, copy `template.env` to `.env`, and set the corresponding values.
Plots and experiment run scripts might require replacing additional variables, marked with `TODO`.

Our code in `src/` follows a standard Python project structure.
Make sure to add the `src/` directory to your `PYTHONPATH`, in order to ensure everything works as intended.

Certain experiments and plots require additional files.
For SELENA, `notebooks/generate_selena_similarities.ipynb` calculates the OpenCLIP embeddings
over all of CIFAR-10, used for similarity measurements.
Furthermore, `generate_ood_imagenet_samples.py` extracts the OOD samples from ImageNet
that we use as SSL canaries.
We include the resulting images in `ood_imagenet_samples.pt` for convenience.

## Running individual defenses
Each experiment in our paper corresponds to an executable Python module in `src/experiments/`.
We provide one module for each heuristic defense in our case studies.
Additionally, `experiments.undefended` trains a model without any defense,
`experiments.approx_worst_case` reproduces the sample-level attacks in Figure 2,
and `experiments.validate_loo` verifies our audit setup.

Each of the above modules accepts various command line arguments;
see their documentation for more details.

## Reproducing full results
We provide scripts to reproduce the full results in our paper in `scripts/`.
For experiment `X`, `submit_X.sh` are Slurm batch scripts to train a set of models.
Then, `attack_X.sh` are normal bash scripts that run the corresponding attack on the results.
To run those files, you need to update the paths marked with `TODO` in each script.

We further provide a Jupyter notebook `notebooks/plot-paper.ipynb` that generates all plots in the paper.
