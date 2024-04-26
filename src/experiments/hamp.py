import argparse
import json
import multiprocessing
import os
import pathlib
import typing

import dotenv
import filelock
import mlflow
import numpy as np
import scipy.optimize
import sklearn.linear_model
import sklearn.metrics
import torch
import torch.utils.data
import torchdata.dataloader2
import torchdata.datapipes.map
import torchvision
import torchvision.transforms.v2
import tqdm

import attack_util
import base
import data
import models


def main():
    dotenv.load_dotenv()
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    experiment_base_dir = args.experiment_dir.expanduser().resolve()

    experiment_name = args.experiment
    run_suffix = args.run_suffix
    verbose = args.verbose

    global_seed = args.seed
    base.setup_seeds(global_seed)

    num_shadow = args.num_shadow
    assert num_shadow > 0
    num_canaries = args.num_canaries
    assert num_canaries > 0
    num_poison = args.num_poison
    assert num_poison >= 0

    data_generator = data.DatasetGenerator(
        num_shadow=num_shadow,
        num_canaries=num_canaries,
        canary_type=data.CanaryType(args.canary_type),
        num_poison=num_poison,
        poison_type=data.PoisonType(args.poison_type),
        data_dir=data_dir,
        seed=global_seed,
        download=bool(os.environ.get("DOWNLOAD_DATA")),
    )
    directory_manager = DirectoryManager(
        experiment_base_dir=experiment_base_dir,
        experiment_name=experiment_name,
        run_suffix=run_suffix,
    )

    if args.action == "attack":
        logreg_c = args.logreg_c
        assert logreg_c > 0.0
        # Attack only depends on global seed (if any)
        _run_attack(args, logreg_c, global_seed, data_generator, directory_manager)
    elif args.action == "train":
        shadow_model_idx = args.shadow_model_idx
        assert 0 <= shadow_model_idx < num_shadow
        setting_seed = base.get_setting_seed(
            global_seed=global_seed, shadow_model_idx=shadow_model_idx, num_shadow=num_shadow
        )
        base.setup_seeds(setting_seed)

        _run_train(
            args,
            shadow_model_idx,
            data_generator,
            directory_manager,
            setting_seed,
            experiment_name,
            run_suffix,
            verbose,
        )
    else:
        assert False, f"Unknown action {args.action}"


def _run_train(
    args: argparse.Namespace,
    shadow_model_idx: int,
    data_generator: data.DatasetGenerator,
    directory_manager: "DirectoryManager",
    training_seed: int,
    experiment_name: str,
    run_suffix: typing.Optional[str],
    verbose: bool,
) -> None:
    rng = np.random.default_rng(seed=training_seed)

    # Hyperparameters
    entropy_threshold = args.entropy_threshold
    assert 0.0 <= entropy_threshold <= 1.0
    regularization_strength = args.regularization_strength
    assert regularization_strength >= 0.0
    num_epochs = 200

    print(
        f"Training shadow model {shadow_model_idx}, entropy threshold {entropy_threshold}, regularization strength {regularization_strength}"
    )
    print(
        f"{data_generator.num_canaries} canaries ({data_generator.canary_type.value}), "
        f"{data_generator.num_poison} poisons ({data_generator.poison_type.value})"
    )

    output_dir = directory_manager.get_training_output_dir(shadow_model_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = directory_manager.get_training_log_dir(shadow_model_idx)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_data = data_generator.build_train_data(
        shadow_model_idx=shadow_model_idx,
    )

    # Make sure only one run creates the MLFlow experiment and starts at a time to avoid concurrency issues
    with filelock.FileLock(log_dir / "enter_mlflow.lock"):
        mlflow.set_tracking_uri(f"file:{log_dir}")
        mlflow.set_experiment(experiment_name=experiment_name)
        run_name = f"train_{shadow_model_idx}"
        if run_suffix is not None:
            run_name += f"_{run_suffix}"
        run = mlflow.start_run(run_name=run_name)
    with run:
        mlflow.log_params(
            {
                "shadow_model_idx": shadow_model_idx,
                "num_canaries": data_generator.num_canaries,
                "canary_type": data_generator.canary_type.value,
                "num_poison": data_generator.num_poison,
                "poison_type": data_generator.poison_type.value,
                "training_seed": training_seed,
                "num_epochs": num_epochs,
                "entropy_threshold": entropy_threshold,
                "regularization_strength": regularization_strength,
            }
        )
        current_model = _train_model(
            train_data,
            training_seed=training_seed,
            num_epochs=num_epochs,
            entropy_threshold=entropy_threshold,
            regularization_strength=regularization_strength,
            verbose=verbose,
        )
        current_model.eval()

        torch.save(current_model, output_dir / "model.pt")
        print("Saved model")

        metrics = dict()

        print("Predicting logits and evaluating full training data")
        full_train_data = data_generator.build_full_train_data()
        train_data_full_pipe = full_train_data.as_unlabeled().build_datapipe()
        # NB: Always predict on augmented samples, even if not training with data augmentation
        rng_pred_train, rng = rng.spawn(2)
        train_pred_full_traintime, train_pred_full_testtime = _predict(
            current_model,
            train_data_full_pipe,
            rng=rng_pred_train,
            data_augmentation=True,
            include_test_time_defense=True,
        )
        del rng_pred_train
        torch.save(train_pred_full_traintime, output_dir / "predictions_train_traintime.pt")
        torch.save(train_pred_full_testtime, output_dir / "predictions_train_testtime.pt")

        train_membership_mask = data_generator.build_in_mask(shadow_model_idx)  # does not include poisons
        # NB: Inference defense does not change label; sufficient to use one set of predictions
        train_ys_pred = torch.argmax(train_pred_full_testtime[:, 0], dim=-1)
        train_ys = full_train_data.targets
        correct_predictions_train = torch.eq(train_ys_pred, train_ys).to(dtype=base.DTYPE_EVAL)
        metrics.update(
            {
                "train_accuracy_full": torch.mean(correct_predictions_train).item(),
                "train_accuracy_in": torch.mean(correct_predictions_train[train_membership_mask]).item(),
                "train_accuracy_out": torch.mean(correct_predictions_train[~train_membership_mask]).item(),
            }
        )
        print(f"Train accuracy (full data): {metrics['train_accuracy_full']:.4f}")
        print(f"Train accuracy (only IN samples): {metrics['train_accuracy_in']:.4f}")
        print(f"Train accuracy (only OUT samples): {metrics['train_accuracy_out']:.4f}")
        canary_mask = torch.zeros_like(train_membership_mask)
        canary_mask[data_generator.get_canary_indices()] = True
        metrics.update(
            {
                "train_accuracy_canaries": torch.mean(correct_predictions_train[canary_mask]).item(),
                "train_accuracy_canaries_in": torch.mean(
                    correct_predictions_train[canary_mask & train_membership_mask]
                ).item(),
                "train_accuracy_canaries_out": torch.mean(
                    correct_predictions_train[canary_mask & (~train_membership_mask)]
                ).item(),
            }
        )
        print(f"Train accuracy (full canary subset): {metrics['train_accuracy_canaries']:.4f}")
        print(f"Train accuracy (IN canary subset): {metrics['train_accuracy_canaries_in']:.4f}")
        print(f"Train accuracy (OUT canary subset): {metrics['train_accuracy_canaries_out']:.4f}")

        print("Evaluating on test data")
        test_metrics, test_pred = _evaluate_model_test(current_model, data_generator)
        metrics.update(test_metrics)
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        torch.save(test_pred, output_dir / "predictions_test.pt")
        mlflow.log_metrics(metrics, step=num_epochs)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)


def _evaluate_model_test(
    model: torch.nn.Module,
    data_generator: data.DatasetGenerator,
    disable_tqdm: bool = False,
) -> typing.Tuple[typing.Dict[str, float], torch.Tensor]:
    test_data = data_generator.build_test_data()
    test_ys = test_data.targets
    test_xs_datapipe = test_data.as_unlabeled().build_datapipe()
    # Test-time defense does not change labels, hence can avoid computational overhead
    test_pred, _ = _predict(
        model,
        test_xs_datapipe,
        data_augmentation=False,
        include_test_time_defense=False,
        rng=None,
        disable_tqdm=disable_tqdm,
    )
    test_ys_pred = torch.argmax(test_pred[:, 0], dim=-1)
    correct_predictions = torch.eq(test_ys_pred, test_ys).to(base.DTYPE_EVAL)
    return {
        "test_accuracy": torch.mean(correct_predictions).item(),
    }, test_pred


def _run_attack(
    args: argparse.Namespace,
    logreg_c: float,
    global_seed: int,
    data_generator: data.DatasetGenerator,
    directory_manager: "DirectoryManager",
) -> None:
    output_dir = directory_manager.get_attack_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_ys, shadow_membership_mask, canary_indices = data_generator.build_attack_data()
    labels_file = output_dir / "attack_ys.pt"
    torch.save(attack_ys, labels_file)
    print(f"Saved audit ys to {labels_file}")

    # Indices of samples with noise (if any)
    canary_indices_file = output_dir / "canary_indices.pt"
    torch.save(canary_indices, canary_indices_file)
    print(f"Saved canary indices to {canary_indices_file}")

    assert shadow_membership_mask.size() == (data_generator.num_raw_training_samples, data_generator.num_shadow)
    membership_file = output_dir / "shadow_membership_mask.pt"
    torch.save(shadow_membership_mask, membership_file)
    print(f"Saved membership to {membership_file}")

    # Attack train-time and test-time defenses individually
    for defense in ("traintime", "testtime"):
        print(f"# Attacking {defense} defense")

        # Load logits
        shadow_logits_raw = []
        for shadow_model_idx in range(data_generator.num_shadow):
            shadow_model_dir = directory_manager.get_training_output_dir(shadow_model_idx)
            shadow_logits_raw.append(torch.load(shadow_model_dir / f"predictions_train_{defense}.pt"))
        shadow_logits = torch.stack(shadow_logits_raw, dim=1)
        del shadow_logits_raw
        assert shadow_logits.dim() == 4  # samples x shadow models x augmentations x classes
        assert shadow_logits.size(0) == data_generator.num_raw_training_samples
        assert shadow_logits.size(1) == data_generator.num_shadow
        num_augmentations = shadow_logits.size(2)

        shadow_scores_full = {
            "hinge": attack_util.hinge_score(shadow_logits, attack_ys),
            "logit": attack_util.logit_score(shadow_logits, attack_ys),
        }
        # Only care about canaries
        shadow_scores = {score_name: scores[canary_indices] for score_name, scores in shadow_scores_full.items()}
        assert all(
            scores.size() == (data_generator.num_raw_training_samples, data_generator.num_shadow, num_augmentations)
            for scores in shadow_scores_full.values()
        )

        # Global threshold
        print("## Global threshold")
        for score_name, scores in shadow_scores.items():
            print(f"### {score_name}")
            # Use score on first data augmentation (= no augmentations)
            # => scores and membership have same size, can just flatten both
            _eval_attack(
                attack_scores=scores[:, :, 0].view(-1),
                attack_membership=shadow_membership_mask[canary_indices].view(-1),
                output_dir=output_dir,
                suffix=f"{defense}_global_{score_name}",
            )
        print()

        # LiRA
        for is_augmented in (False, True):
            print(f"## LiRA {'w/' if is_augmented else 'w/o'} data augmentation")
            attack_suffix = "lira_da" if is_augmented else "lira"
            if is_augmented:
                shadow_attack_data = {
                    score_name: attack_util.lira_attack_loo(
                        shadow_scores=scores,
                        shadow_membership_mask=shadow_membership_mask[canary_indices],
                    )
                    for score_name, scores in shadow_scores.items()
                }
            else:
                shadow_attack_data = {
                    score_name: attack_util.lira_attack_loo(
                        shadow_scores=scores[:, :, 0].unsqueeze(-1),
                        shadow_membership_mask=shadow_membership_mask[canary_indices],
                    )
                    for score_name, scores in shadow_scores.items()
                }

            for score_name, (scores, membership) in shadow_attack_data.items():
                print(f"### {score_name}")
                _eval_attack(
                    attack_scores=scores,
                    attack_membership=membership,
                    output_dir=output_dir,
                    suffix=f"{defense}_{attack_suffix}_{score_name}",
                )

            print()

        # Label-only; can do for either defenses b/c predicted labels remain the same
        if defense == "testtime":
            print("## Label-only attack")
            all_labelonly_xs = (
                (shadow_logits[canary_indices].argmax(-1) == attack_ys[canary_indices].view(-1, 1, 1))
                .numpy()
                .astype(attack_util.LIRA_DTYPE_NUMPY)
            )
            all_labelonly_ys = shadow_membership_mask[canary_indices].numpy().astype(int)
            num_canaries = all_labelonly_xs.shape[0]

            with multiprocessing.Pool(16) as pool:
                for hp_name, logreg_c_value in (("tuned", logreg_c), ("default", 1.0)):
                    print(f"### {hp_name} hyperparameters")

                    # Leave one out
                    labelonly_scores_raw = np.empty(
                        (num_canaries, data_generator.num_shadow), dtype=attack_util.LIRA_DTYPE_NUMPY
                    )
                    for target_model_idx in range(data_generator.num_shadow):
                        current_attack = _LabelOnlyLogisticRegressionAttack(
                            all_membership=all_labelonly_ys,
                            all_features=all_labelonly_xs,
                            target_model_idx=target_model_idx,
                            logreg_C=logreg_c_value,
                            seed=global_seed,
                        )
                        for sample_idx, sample_attack_score in enumerate(
                            pool.imap(current_attack, range(num_canaries), chunksize=16)
                        ):
                            labelonly_scores_raw[sample_idx, target_model_idx] = sample_attack_score

                    labelonly_membership = torch.from_numpy(all_labelonly_ys.flatten())
                    labelonly_attack_scores = torch.from_numpy(labelonly_scores_raw.flatten()).to(
                        dtype=attack_util.LIRA_DTYPE_TORCH
                    )

                    _eval_attack(
                        attack_scores=labelonly_attack_scores,
                        attack_membership=labelonly_membership,
                        output_dir=output_dir,
                        suffix=f"labelonly_{hp_name}",
                    )
            print()

        # To avoid memory congestion issues
        del shadow_logits
        print()


def _eval_attack(
    attack_scores: torch.Tensor,
    attack_membership: torch.Tensor,
    output_dir: pathlib.Path,
    suffix: str = "",
) -> None:
    score_file = output_dir / f"attack_scores_{suffix}.pt"
    torch.save(attack_scores, score_file)
    membership_file = output_dir / f"attack_membership_{suffix}.pt"
    torch.save(attack_membership, membership_file)

    # Calculate TPR at various FPR
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=attack_membership.int().numpy(), y_score=attack_scores.numpy())
    target_fprs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
    for target_fpr in target_fprs:
        print(f"TPR at FPR {target_fpr*100}%: {tpr[fpr <= target_fpr][-1]*100:.4f}%")

    # Calculate attack accuracy
    prediction_threshold = torch.median(attack_scores).item()
    pred_membership = attack_scores > prediction_threshold  # median returns lower of two values => strict ineq.
    balanced_accuracy = torch.mean((pred_membership == attack_membership).float()).item()
    print(f"Attack accuracy: {balanced_accuracy:.4f}")


def _train_model(
    train_data: data.Dataset,
    training_seed: int,
    num_epochs: int,
    entropy_threshold: float,
    regularization_strength: float,
    verbose: bool = False,
) -> torch.nn.Module:
    # Hyperparameters from original paper
    batch_size = 64
    momentum = 0.99
    weight_decay = 1e-5
    learning_rate = 0.5

    num_classes = 10

    # NB: Original code uses ResNet-18; use WRN16-4 here for consistency w/ other experiments
    model = models.WideResNet(
        in_channels=3,
        depth=16,
        widen_factor=4,
        num_classes=num_classes,
        use_group_norm=False,
        device=base.DEVICE,
        dtype=base.DTYPE,
    )

    train_datapipe = train_data.build_datapipe(
        shuffle=True,
        add_sharding_filter=True,
    )
    # Calculate target/other class probabilities for high-entropy soft labels
    prob_target, prob_other = _calculate_soft_label_probs(entropy_threshold, num_classes=num_classes)
    print(f"Using soft label probabilities {prob_target=:.4f}, {prob_other=:.4f}")

    train_datapipe = train_datapipe.map(
        _ApplySoftLabels(
            prob_target=prob_target,
            prob_other=prob_other,
            num_classes=num_classes,
        )
    )

    train_datapipe = train_datapipe.map(
        torchvision.transforms.v2.Compose(
            [
                # Original paper does NOT use data augmentation
                # torchvision.transforms.v2.RandomCrop(32, padding=4),
                # torchvision.transforms.v2.RandomHorizontalFlip(),
                torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
                torchvision.transforms.v2.Normalize(
                    mean=data.CIFAR10_MEAN,
                    std=data.CIFAR10_STD,
                ),
            ]
        ),
        input_col=0,
        output_col=0,
    )
    train_datapipe = train_datapipe.batch(batch_size, drop_last=False).collate()

    train_loader = torchdata.dataloader2.DataLoader2(
        train_datapipe,
        reading_service=torchdata.dataloader2.MultiProcessingReadingService(
            num_workers=4,
        ),
    )

    # his is a bit of a hack to ensure that the # steps per epoch is correct
    #  Using the length of the training datapipe might be wrong due to sharding and multiprocessing
    num_steps_per_epoch = 0
    for _ in train_loader:
        num_steps_per_epoch += 1

    # NB: Original code resets optimizer every epoch; we keep the same (typo in original code?)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Same lr schedule as in original paper
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 90, 150],
        gamma=0.1,
    )

    model.train()
    rng_loader_seeds = np.random.default_rng(seed=training_seed)
    for epoch in (pbar := tqdm.trange(num_epochs, desc="Training", unit="epoch")):
        num_samples = 0
        epoch_loss = 0.0
        epoch_loss_kl = 0.0
        epoch_loss_entropy = 0.0
        epoch_accuracy = 0.0
        train_loader.seed(int(rng_loader_seeds.integers(0, 2**32, dtype=np.uint32, size=())))
        for batch_xs, batch_targets in tqdm.tqdm(
            train_loader,
            desc="Current epoch",
            unit="batch",
            leave=False,
            disable=not verbose,
            total=num_steps_per_epoch,
        ):
            batch_xs = batch_xs.to(base.DEVICE)
            batch_targets = batch_targets.to(base.DEVICE)

            optimizer.zero_grad()
            batch_pred = model(batch_xs)

            batch_log_probs = torch.nn.functional.log_softmax(batch_pred, dim=1)
            # NB: Original code use KL-divergence loss, but in a wrong way (wrong mult. constant b/c reduction `mean`)
            #  Hence, we have to do the same, else the HAMP hyperparameters do not match...
            batch_loss_kl = torch.nn.functional.kl_div(
                input=batch_log_probs,
                target=batch_targets,
                log_target=False,
                reduction="mean",  # technically incorrect usage of loss function
            )
            batch_entropy = -torch.sum(batch_log_probs.exp() * batch_log_probs, dim=1)
            batch_loss_entropy = regularization_strength * torch.mean(batch_entropy)
            batch_loss = batch_loss_kl - batch_loss_entropy

            batch_loss.backward()
            optimizer.step()

            # Targets are soft labels; convert to hard labels for evaluation
            batch_ys = batch_targets.argmax(-1)

            epoch_loss += batch_loss.item() * batch_xs.size(0)
            epoch_loss_kl += batch_loss_kl.item() * batch_xs.size(0)
            epoch_loss_entropy += batch_loss_entropy.item() * batch_xs.size(0)
            epoch_accuracy += (batch_pred.argmax(-1) == batch_ys).int().sum().item()
            num_samples += batch_xs.size(0)
        epoch_loss /= num_samples
        epoch_loss_kl /= num_samples
        epoch_loss_entropy /= num_samples
        epoch_accuracy /= num_samples
        progress_dict = {
            "epoch_loss": epoch_loss,
            "epoch_loss_kl": epoch_loss_kl,
            "epoch_loss_entropy": epoch_loss_entropy,
            "epoch_accuracy": epoch_accuracy,
            "lr": lr_scheduler.get_last_lr()[0],
        }

        mlflow.log_metrics(progress_dict, step=epoch + 1)
        pbar.set_postfix(progress_dict)

        # Lr schedule is per epoch
        lr_scheduler.step()

    model.eval()
    return model


def _predict(
    model: torch.nn.Module,
    datapipe: torchdata.datapipes.iter.IterDataPipe,
    rng: typing.Optional[np.random.Generator],
    data_augmentation: bool,
    include_test_time_defense: bool,
    disable_tqdm: bool = False,
) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
    if include_test_time_defense:
        assert rng is not None

    # NB: Always returns data-augmentation dimension
    model.eval()
    dtype_transform = torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True)
    datapipe = datapipe.map(dtype_transform)
    normalize_transform = torchvision.transforms.v2.Normalize(
        mean=data.CIFAR10_MEAN,
        std=data.CIFAR10_STD,
    )
    if not data_augmentation:
        # Augmentations add normalization later
        datapipe = datapipe.map(normalize_transform)
    datapipe = datapipe.batch(base.EVAL_BATCH_SIZE, drop_last=False).collate()

    random_samples_transform = torchvision.transforms.v2.Compose(
        [
            dtype_transform,
            normalize_transform,
        ]
    )

    # First, do all actual predictions w/ only training-time defense, and apply test-time defense for all at once
    pred_logits_traintime = []
    pred_logits_random = []
    with torch.no_grad():
        for batch_xs in tqdm.tqdm(datapipe, desc="Predicting", unit="batch", disable=disable_tqdm):
            if not data_augmentation:
                pred_logits_traintime.append(
                    model(batch_xs.to(dtype=base.DTYPE, device=base.DEVICE)).cpu().unsqueeze(1)
                )
                if include_test_time_defense:
                    pred_logits_random.append(
                        _predict_random_images(model, batch_xs.size(), rng, random_samples_transform).unsqueeze(1)
                    )
            else:
                flip_augmentations = (False, True)
                shift_augmentations = (0, -4, 4)
                batch_xs_pad = torchvision.transforms.v2.functional.pad(
                    batch_xs,
                    padding=[4],
                )
                pred_logits_current = []
                pred_logits_random_current = []
                for flip in flip_augmentations:
                    for shift_y in shift_augmentations:
                        for shift_x in shift_augmentations:
                            offset_y = shift_y + 4
                            offset_x = shift_x + 4
                            batch_xs_aug = batch_xs_pad[:, :, offset_y : offset_y + 32, offset_x : offset_x + 32]
                            if flip:
                                batch_xs_aug = torchvision.transforms.v2.functional.hflip(batch_xs_aug)
                            # Normalization did not happen before; do it here
                            batch_xs_aug = normalize_transform(batch_xs_aug)
                            pred_logits_current.append(
                                model(batch_xs_aug.to(dtype=base.DTYPE, device=base.DEVICE)).cpu()
                            )

                            if include_test_time_defense:
                                pred_logits_random_current.append(
                                    _predict_random_images(model, batch_xs.size(), rng, random_samples_transform)
                                )

                pred_logits_traintime.append(torch.stack(pred_logits_current, dim=1))
                if include_test_time_defense:
                    pred_logits_random.append(torch.stack(pred_logits_random_current, dim=1))

    pred_logits_traintime = torch.cat(pred_logits_traintime, dim=0)
    if not include_test_time_defense:
        return pred_logits_traintime, None

    # Apply test-time defense
    pred_logits_random = torch.sort(torch.cat(pred_logits_random, dim=0), stable=True, dim=-1).values
    assert pred_logits_random.dtype == pred_logits_random.dtype
    assert torch.all(pred_logits_random.max(-1).values == pred_logits_random[..., -1])

    # Make sure maxima of raw and random logits are unique per-sample to ensure test-time defense does not change label
    # NB: This does not necessarily preserve top-k order for k > 1, but we only care about top-1 here
    pred_logits_random[..., -1] += torch.finfo(pred_logits_random.dtype).eps  # min. pos. value s.t. x + eps > x
    assert torch.all(pred_logits_random[..., -1].unsqueeze(-1) > pred_logits_random[..., :-1])
    pred_labels_traintime = pred_logits_traintime.argmax(-1, keepdim=True)
    pred_logits_traintime.scatter_add_(
        -1,
        index=pred_labels_traintime,
        src=torch.tensor(torch.finfo(pred_logits_traintime.dtype).eps, dtype=pred_logits_traintime.dtype).expand(
            pred_labels_traintime.size()
        ),
    )
    num_classes = pred_logits_traintime.size(-1)
    assert torch.all(
        (pred_logits_traintime.max(-1, keepdim=True).values > pred_logits_traintime).int().sum(-1) == num_classes - 1
    )

    # Calculate defended predictions by reordering random logits
    pred_label_order = torch.argsort(pred_logits_traintime, stable=True, dim=-1)
    pred_logits_testtime = torch.empty_like(pred_logits_random)
    pred_logits_testtime.scatter_(dim=-1, index=pred_label_order, src=pred_logits_random)
    assert pred_logits_testtime.size() == pred_logits_traintime.size()
    # This should always be true since both the maxima of raw and random logits are unique per-sample
    assert torch.all(pred_logits_testtime.argmax(-1) == pred_logits_traintime.argmax(-1))

    return pred_logits_traintime, pred_logits_testtime


def _calculate_soft_label_probs(entropy_threshold: float, num_classes: int) -> typing.Tuple[float, float]:
    # FIXME: Can we get this in closed form?

    def f(p: float) -> float:
        return -(p * np.log(p) + (1 - p) * np.log((1 - p) / (num_classes - 1))) - entropy_threshold * np.log(
            num_classes
        )

    # FIXME: The resulting threshold could technically be slightly above threshold (up to tolerance); does not matter
    # p should always be at least 1/k (max entropy), and less than 1.0
    p_res = scipy.optimize.root_scalar(f, bracket=(1.0 / num_classes, 1.0 - 1e-12))
    assert p_res.converged
    p = p_res.root
    return p, (1 - p) / (num_classes - 1)


class _LabelOnlyLogisticRegressionAttack(typing.Callable[[int], float]):
    def __init__(
        self,
        all_membership: np.ndarray,
        all_features: np.ndarray,
        target_model_idx: int,
        logreg_C: float,
        seed: int,
    ):
        self.all_membership = all_membership
        self.all_features = all_features
        self.target_model_idx = target_model_idx
        self.logreg_C = logreg_C
        self.seed = seed

    def __call__(self, sample_idx: int) -> float:
        # ys are membership
        train_ys = np.delete(self.all_membership[sample_idx], self.target_model_idx, axis=0)

        # xs are correct predictions over data augmentations
        train_xs = np.delete(self.all_features[sample_idx], self.target_model_idx, axis=0)
        test_xs = self.all_features[sample_idx, self.target_model_idx].reshape(1, -1)

        regressor = sklearn.linear_model.LogisticRegression(
            C=self.logreg_C,
            penalty="l2",
            random_state=self.seed,
            warm_start=False,
            max_iter=1000,
        )
        regressor.fit(train_xs, train_ys)
        assert regressor.classes_.shape == (2,)
        assert regressor.classes_[0] == 0 and regressor.classes_[1] == 1

        return regressor.predict_proba(test_xs)[0, 1]  # first sample, IN


class _ApplySoftLabels(typing.Callable):
    def __init__(
        self,
        prob_target: float,
        prob_other: float,
        num_classes: int,
    ) -> None:
        self._prob_target = prob_target
        self._prob_other = prob_other
        self._num_classes = num_classes

    def __call__(self, inputs: typing.Tuple[torch.Tensor, torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        features, hard_label = inputs
        soft_labels = torch.ones(self._num_classes) * self._prob_other
        soft_labels[hard_label] = self._prob_target
        return features, soft_labels


def _predict_random_images(
    model: torch.nn.Module,
    batch_shape: torch.Size,
    rng: np.random.Generator,
    transform: torchvision.transforms.v2.Transform,
) -> torch.Tensor:
    # Generate random samples
    # NB: Done iteratively for deterministic results independent of eval batch size
    batch_xs_random = []
    for _ in range(batch_shape[0]):
        batch_xs_random.append(rng.integers(0, 256, size=batch_shape[1:], dtype=np.uint8))
    batch_xs_random = torch.from_numpy(np.stack(batch_xs_random))
    assert batch_xs_random.size() == batch_shape

    # Predict on random samples
    return model(transform(batch_xs_random).to(dtype=base.DTYPE, device=base.DEVICE)).cpu()


class DirectoryManager(object):
    def __init__(
        self,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
        run_suffix: typing.Optional[str] = None,
    ) -> None:
        self._experiment_base_dir = experiment_base_dir
        self._experiment_dir = self._experiment_base_dir / experiment_name
        self._run_suffix = run_suffix

    def get_training_output_dir(self, shadow_model_idx: typing.Optional[int]) -> pathlib.Path:
        actual_suffix = "" if self._run_suffix is None else f"_{self._run_suffix}"
        return self._experiment_dir / ("shadow" + actual_suffix) / str(shadow_model_idx)

    def get_training_log_dir(self, shadow_model_idx: typing.Optional[int]) -> pathlib.Path:
        # Log all MLFlow stuff into the same directory, for all experiments!
        return self._experiment_base_dir / "mlruns"

    def get_attack_output_dir(self) -> pathlib.Path:
        return self._experiment_dir if self._run_suffix is None else self._experiment_dir / f"attack_{self._run_suffix}"


def parse_args() -> argparse.Namespace:
    default_data_dir = pathlib.Path(os.environ.get("DATA_ROOT", "data"))
    default_base_experiment_dir = pathlib.Path(os.environ.get("EXPERIMENT_DIR", ""))

    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--data-dir", type=pathlib.Path, default=default_data_dir, help="Dataset root directory")
    parser.add_argument(
        "--experiment-dir", default=default_base_experiment_dir, type=pathlib.Path, help="Experiment directory"
    )
    parser.add_argument("--experiment", type=str, default="dev", help="Experiment name")
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional run suffix to distinguish multiple runs in the same experiment",
    )
    parser.add_argument("--verbose", action="store_true")

    # Dataset and setup args
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-shadow", type=int, default=64, help="Number of shadow models")
    parser.add_argument("--num-canaries", type=int, default=500, help="Number of canaries to audit")
    parser.add_argument(
        "--canary-type",
        type=data.CanaryType,
        default=data.CanaryType.CLEAN,
        choices=list(data.CanaryType),
        help="Type of canary to use",
    )
    parser.add_argument("--num-poison", type=int, default=0, help="Number of poison samples to include")
    parser.add_argument(
        "--poison-type",
        type=data.PoisonType,
        default=data.PoisonType.CANARY_DUPLICATES,
        choices=list(data.PoisonType),
        help="Type of poisoning to use",
    )

    # Create subparsers per action
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--shadow-model-idx", type=int, required=True, help="Train shadow model with index if present"
    )

    # Defense-specific
    train_parser.add_argument("--entropy-threshold", type=float, default=0.95, help="Entropy threshold (gamma)")
    train_parser.add_argument(
        "--regularization-strength", type=float, default=0.001, help="Regularization strength (alpha)"
    )

    # MIA
    attack_parser = subparsers.add_parser("attack")

    # Defense-specific
    attack_parser.add_argument(
        "--logreg-c", type=float, required=True, help="Logistic regression regularization parameter"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
