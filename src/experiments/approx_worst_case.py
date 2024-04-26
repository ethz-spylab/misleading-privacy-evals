import argparse
import gc
import hashlib
import pathlib
import typing

import dotenv
import numpy as np
import scipy.stats
import sklearn.metrics
import tqdm

import base

KEEP_MD5 = "77040c727b8668156982637da2b82fe0"
SCORE_MD5 = "7ec9b03437437052ab5cc97398868089"

EPS = 1e-30
TARGET_FPR = 0.001
TOP_VULNERABLE_K = 500  # number of top vulnerable samples to calculate ROC over


def main():
    dotenv.load_dotenv()
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()

    seed = args.seed
    base.setup_seeds(seed)
    rng = np.random.default_rng(seed=seed)

    num_shadow = args.num_shadow
    assert num_shadow > 0

    print("Loading 20k models scores")
    keep_path = experiment_dir / "keep_20k.npy"
    score_path = experiment_dir / "score_20k.npy"
    keep_hash = hashlib.md5(keep_path.read_bytes()).hexdigest()
    score_hash = hashlib.md5(score_path.read_bytes()).hexdigest()
    if keep_hash != KEEP_MD5:
        raise ValueError(f"{keep_path} has wrong hash {keep_hash}, expected {KEEP_MD5}")
    if keep_hash != KEEP_MD5:
        raise ValueError(f"{score_path} has wrong hash {score_hash}, expected {SCORE_MD5}")
    keep = np.load(keep_path)
    score = np.load(score_path)
    num_models, num_samples = keep.shape
    assert score.shape == (num_models, num_samples)

    num_targets = num_models - num_shadow
    shadow_indices = rng.choice(num_models, size=num_shadow, replace=False)
    target_indices = np.setdiff1d(np.arange(num_models), shadow_indices)
    shadow_keep = keep[shadow_indices]
    shadow_score = score[shadow_indices]
    print("Average shadow model membership:", shadow_keep.sum(axis=0).mean())

    means_in = np.empty((num_samples,), dtype=np.float64)
    means_out = np.empty_like(means_in)
    stds_in = np.empty_like(means_in)
    stds_out = np.empty_like(means_in)
    for sample_idx in tqdm.trange(num_samples, desc="Fitting Gaussians on shadow model predictions"):
        scores_in = shadow_score[shadow_keep[:, sample_idx], sample_idx]
        scores_out = shadow_score[~shadow_keep[:, sample_idx], sample_idx]
        means_in[sample_idx] = np.mean(scores_in, axis=0)
        means_out[sample_idx] = np.mean(scores_out, axis=0)
        stds_in[sample_idx] = np.std(scores_in, axis=0) + EPS
        stds_out[sample_idx] = np.std(scores_out, axis=0) + EPS

    target_keep = keep[target_indices]
    target_score = score[target_indices]
    attack_scores_targets = np.empty((num_samples, num_targets), dtype=np.float64)
    for sample_idx in tqdm.trange(num_samples, desc="Attacking targets"):
        target_scores = target_score[:, sample_idx]
        mean_in = means_in[sample_idx]
        mean_out = means_out[sample_idx]
        std_in = stds_in[sample_idx]
        std_out = stds_out[sample_idx]

        log_prs_in = scipy.stats.norm.logpdf(target_scores, mean_in, std_in)
        log_prs_out = scipy.stats.norm.logpdf(target_scores, mean_out, std_out)
        attack_scores_targets[sample_idx] = log_prs_in - log_prs_out

    del keep, score, shadow_keep, shadow_score, target_score
    gc.collect()
    mp_helper = ROCFunctionsForMultiprocessing(target_keep, attack_scores_targets)

    tpr_at_fpr = np.empty((num_samples,), dtype=np.float64)
    for sample_idx, current_tpr_at_fpr in tqdm.tqdm(
        enumerate(map(mp_helper.tpr_at_fpr_for_sample, range(num_samples))),
        total=num_samples,
        desc="Computing per-sample TPR at low FPR",
    ):
        tpr_at_fpr[sample_idx] = current_tpr_at_fpr
    tpr_file = experiment_dir / "per_sample_tpr.npy"
    np.save(tpr_file, tpr_at_fpr)
    print("Saved per-sample TPR at low FPR to", tpr_file)

    max_tpr_sample_idx = np.argmax(tpr_at_fpr)
    print(f"Calculating ROC curve for sample with highest TPR@0.1% FPR (index {max_tpr_sample_idx})")
    max_fprs, max_tprs, _ = sklearn.metrics.roc_curve(
        target_keep[:, max_tpr_sample_idx],
        attack_scores_targets[max_tpr_sample_idx],
    )
    max_roc_file = experiment_dir / "most_vulnerable_roc.npz"
    np.savez(max_roc_file, fpr=max_fprs, tpr=max_tprs)
    print("Saved ROC curve for most vulnerable sample to", max_roc_file)
    del max_fprs, max_tprs

    print(f"Calculating ROC curve for top-{TOP_VULNERABLE_K} samples with highest TPR@0.1% FPR")
    top_k_tpr_sample_indices = np.argsort(tpr_at_fpr)[-TOP_VULNERABLE_K:]
    top_k_fprs, top_k_tprs, _ = sklearn.metrics.roc_curve(
        target_keep[:, top_k_tpr_sample_indices].flatten(),
        attack_scores_targets[top_k_tpr_sample_indices].T.flatten(),
    )
    top_k_roc_file = experiment_dir / "topk_vulnerable_roc.npz"
    np.savez(top_k_roc_file, fpr=top_k_fprs, tpr=top_k_tprs)
    print("Saved ROC curve for top vulnerable samples to", top_k_roc_file)
    del tpr_at_fpr, top_k_fprs, top_k_tprs
    gc.collect()

    # Calculate mean ROC curve
    unified_fpr = np.linspace(0, 1, 25000)
    unified_tprs = []
    for fpr, tpr in tqdm.tqdm(
        map(mp_helper.roc_for_target, range(num_targets)), total=num_targets, desc="Averaging per-target ROC curves"
    ):
        unified_tprs.append(np.interp(unified_fpr, fpr, tpr))

    tpr_mean = np.mean(unified_tprs, axis=0)
    tpr_sem = np.std(unified_tprs, axis=0) / np.sqrt(len(unified_tprs))
    mean_roc_file = experiment_dir / "mean_roc.npz"
    np.savez(mean_roc_file, fpr=unified_fpr, tpr_mean=tpr_mean, tpr_sem=tpr_sem)
    print("Saved mean ROC curve to", mean_roc_file)

    print(
        f"Averaged TPR@0.1% FPR: {max(tpr_mean[unified_fpr <= TARGET_FPR])*100:.2f}% (±{tpr_sem[unified_fpr <= TARGET_FPR][0]*100:.2f})"
    )


class ROCFunctionsForMultiprocessing(object):
    def __init__(self, target_keep: np.ndarray, attack_scores_targets: np.ndarray):
        self.target_keep = target_keep
        self.attack_scores_targets = attack_scores_targets

    def tpr_at_fpr_for_sample(self, sample_idx: int):
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            self.target_keep[:, sample_idx],
            self.attack_scores_targets[sample_idx],
        )
        return max(tpr[fpr <= TARGET_FPR])

    def roc_for_target(self, target_idx: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            self.target_keep[target_idx],
            self.attack_scores_targets[:, target_idx],
            drop_intermediate=False,
        )
        return fpr, tpr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--experiment-dir", required=True, type=pathlib.Path, help="Experiment directory")

    # Dataset and setup args
    parser.add_argument("--seed", type=int, default=0xCAFE)
    parser.add_argument("--num-shadow", type=int, default=64, help="Number of shadow models within the 20k models")

    return parser.parse_args()


if __name__ == "__main__":
    main()
