import argparse
import json
import os
import pathlib
import typing
import warnings

import dotenv
import filelock
import mlflow
import numpy as np
import opacus
import opacus.utils.batch_memory_manager
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
import dpsgd_utils
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
        # Attack only depends on global seed (if any)
        _run_attack(args, data_generator, directory_manager)
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
    # Hyperparameters
    num_epochs = args.num_epochs
    noise_multiplier = args.noise_multiplier
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    augmult_factor = args.augmult_factor
    pretrain_epochs = args.pretraining_epochs

    print(f"Training shadow model {shadow_model_idx}")
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
                "noise_multiplier": noise_multiplier,
                "max_grad_norm": max_grad_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "augmult_factor": augmult_factor,
            }
        )
        current_model = _train_model(
            train_data,
            data_generator=data_generator,
            training_seed=training_seed,
            num_epochs=num_epochs,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            batch_size=batch_size,
            augmult_factor=augmult_factor,
            pretrain_epochs=pretrain_epochs,
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
        train_pred_full = _predict(current_model, train_data_full_pipe, data_augmentation=True)
        torch.save(train_pred_full, output_dir / "predictions_train.pt")

        train_membership_mask = data_generator.build_in_mask(shadow_model_idx)  # does not include poisons
        train_ys_pred = torch.argmax(train_pred_full[:, 0], dim=-1)
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
    test_pred = _predict(model, test_xs_datapipe, data_augmentation=False, disable_tqdm=disable_tqdm)
    test_ys_pred = torch.argmax(test_pred[:, 0], dim=-1)
    correct_predictions = torch.eq(test_ys_pred, test_ys).to(base.DTYPE_EVAL)
    return {
        "test_accuracy": torch.mean(correct_predictions).item(),
    }, test_pred


def _run_attack(
    args: argparse.Namespace,
    data_generator: data.DatasetGenerator,
    directory_manager: "DirectoryManager",
) -> None:
    global_variance = args.global_variance

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

    # Load logits
    shadow_logits_raw = []
    for shadow_model_idx in range(data_generator.num_shadow):
        shadow_model_dir = directory_manager.get_training_output_dir(shadow_model_idx)
        shadow_logits_raw.append(torch.load(shadow_model_dir / "predictions_train.pt"))
    shadow_logits = torch.stack(shadow_logits_raw, dim=1)
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
    print("# Global threshold")
    print("## on all samples")
    for score_name, scores in shadow_scores.items():
        print(f"## {score_name}")
        # Use score on first data augmentation (= no augmentations)
        # => scores and membership have same size, can just flatten both
        _eval_attack(
            attack_scores=scores[:, :, 0].view(-1),
            attack_membership=shadow_membership_mask[canary_indices].view(-1),
            output_dir=output_dir,
            suffix=f"global_{score_name}",
        )
    print()

    # LiRA
    for is_augmented in (False, True):
        print(f"# LiRA {'w/' if is_augmented else 'w/o'} data augmentation")
        attack_suffix = "lira_da" if is_augmented else "lira"
        if is_augmented:
            shadow_attack_data = {
                score_name: attack_util.lira_attack_loo(
                    shadow_scores=scores,
                    shadow_membership_mask=shadow_membership_mask[canary_indices],
                    global_variance=global_variance,
                )
                for score_name, scores in shadow_scores.items()
            }
        else:
            shadow_attack_data = {
                score_name: attack_util.lira_attack_loo(
                    shadow_scores=scores[:, :, 0].unsqueeze(-1),
                    shadow_membership_mask=shadow_membership_mask[canary_indices],
                    global_variance=global_variance,
                )
                for score_name, scores in shadow_scores.items()
            }

        for score_name, (scores, membership) in shadow_attack_data.items():
            print(f"## {score_name}")
            _eval_attack(
                attack_scores=scores,
                attack_membership=membership,
                output_dir=output_dir,
                suffix=f"{attack_suffix}_{score_name}",
            )

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
    data_generator: data.DatasetGenerator,
    training_seed: int,
    num_epochs: int,
    noise_multiplier: float,
    max_grad_norm: float,
    learning_rate: float,
    batch_size: int,
    augmult_factor: int,
    pretrain_epochs: typing.Optional[int],
    verbose: bool = False,
) -> torch.nn.Module:
    momentum = 0
    weight_decay = 0

    if pretrain_epochs is not None:
        # NB: We never do pretraining in the paper
        model, pretrain_metrics = _pretrain(pretrain_epochs, data_generator, training_seed, verbose)
        mlflow.log_metrics(pretrain_metrics, step=0)
    else:
        model = _build_model(num_classes=10)
    model.train()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Secure RNG turned off", module="dpsgd_utils")
        privacy_engine = dpsgd_utils.PrivacyEngineAugmented(opacus.GradSampleModule.GRAD_SAMPLERS)

    train_datapipe = train_data.build_map_datapipe()

    # FIXME: Theoretically, the use of global mean-std normalization incurs a privacy cost that is unaccounted.
    #  However, the normalization constants are w.r.t. the full CIFAR10 data
    #  and I guess glancing over that fact here is not a big issue.
    # NB: This always aplies normalization before potential data augmentation
    train_datapipe = train_datapipe.map(
        torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
                torchvision.transforms.v2.Normalize(
                    mean=data.CIFAR10_MEAN,
                    std=data.CIFAR10_STD,
                ),
            ]
        ),
    )
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(
        train_datapipe,
        drop_last=False,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    dp_delta = 1e-5
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,  # guarantees proper privacy, but could also disable for simplicity
        K=augmult_factor,
        loss_reduction=loss.reduction,
    )

    # Prepare augmult and exponential moving average as in tan code
    # Need to always use custom hooks, else the modified privacy engine will not work with augmult 0
    augmentation = dpsgd_utils.AugmentationMultiplicity(augmult_factor if augmult_factor > 0 else 1)
    model.GRAD_SAMPLERS[torch.nn.modules.conv.Conv2d] = augmentation.augmented_compute_conv_grad_sample
    model.GRAD_SAMPLERS[torch.nn.modules.linear.Linear] = augmentation.augmented_compute_linear_grad_sample
    model.GRAD_SAMPLERS[torch.nn.GroupNorm] = augmentation.augmented_compute_group_norm_grad_sample
    ema_model = dpsgd_utils.create_ema(model)

    num_true_updates = 0
    for epoch in (pbar := tqdm.trange(num_epochs, desc="Training", unit="epoch")):
        model.train()
        with opacus.utils.batch_memory_manager.BatchMemoryManager(
            data_loader=train_loader,
            # 64 is max that always fits into 3090 VRAM w/o memory fragmentation issues for 16 augmult factor
            # Using 128 does not seem to significantly improve performance
            max_physical_batch_size=64,
            optimizer=optimizer,
        ) as optimized_train_loader:
            num_samples = 0
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for batch_xs, batch_ys in tqdm.tqdm(
                optimized_train_loader,
                desc="Current epoch",
                unit="batch",
                leave=False,
                disable=not verbose,
            ):
                batch_xs = batch_xs.to(base.DEVICE)
                batch_ys = batch_ys.to(base.DEVICE)
                original_batch_size = batch_xs.size(0)

                optimizer.zero_grad(set_to_none=True)
                if augmult_factor > 0:
                    # Taken from tan code:
                    batch_xs = torch.repeat_interleave(batch_xs, repeats=augmult_factor, dim=0)
                    batch_ys = torch.repeat_interleave(batch_ys, repeats=augmult_factor, dim=0)
                    transform = torchvision.transforms.v2.Compose(
                        [
                            torchvision.transforms.v2.RandomCrop(32, padding=4),
                            torchvision.transforms.v2.RandomHorizontalFlip(),
                        ]
                    )
                    batch_xs = torchvision.transforms.v2.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
                        batch_xs
                    )
                    assert batch_xs.size(0) == augmult_factor * original_batch_size

                batch_pred = model(batch_xs)
                batch_loss = loss(input=batch_pred, target=batch_ys)
                batch_loss.backward()

                # Only update ema at the end of actual batches
                will_update = not optimizer._check_skip_next_step(pop_next=False)
                optimizer.step()
                if will_update:
                    num_true_updates += 1
                    dpsgd_utils.update_ema(model, ema_model, num_true_updates)

                batch_unnormalized_accuracy = (batch_pred.detach().argmax(-1) == batch_ys).int().sum().item()
                epoch_loss += batch_loss.item() * batch_xs.size(0)
                epoch_accuracy += batch_unnormalized_accuracy
                num_samples += batch_xs.size(0)
            epoch_loss /= num_samples
            epoch_accuracy /= num_samples
            dp_eps = privacy_engine.get_epsilon(dp_delta)
            progress_dict = {
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_accuracy,
                "dp_eps": dp_eps,
                "dp_delta": dp_delta,
                "update_steps": num_true_updates,
            }

            mlflow.log_metrics(progress_dict, step=epoch + 1)
            pbar.set_postfix(progress_dict)

    ema_model.eval()
    return ema_model


def _pretrain(
    pretrain_epochs: int, data_generator: data.DatasetGenerator, training_seed: int, verbose: bool
) -> typing.Tuple[models.WideResNet, typing.Dict[str, float]]:
    # HPs from DeepMind paper (for CIFAR-100 and slightly different network)
    # NB: Default pretrain_epochs would be 200
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    print("Pre-training model on CIFAR-100")
    model = _build_model(num_classes=100)

    train_data = data_generator.load_cifar100()

    indices = tuple(range(len(train_data)))
    train_datapipe = torchdata.datapipes.map.SequenceWrapper(indices, deepcopy=False)
    train_datapipe = train_datapipe.shuffle()
    train_datapipe = train_datapipe.sharding_filter()
    train_datapipe = train_datapipe.map(train_data.__getitem__)
    train_datapipe = train_datapipe.map(
        torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.RandomCrop(32, padding=4),
                torchvision.transforms.v2.RandomHorizontalFlip(),
                torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
                torchvision.transforms.v2.Normalize(
                    mean=(0.5071, 0.4865, 0.4409),
                    std=(0.2008, 0.1983, 0.2002),
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

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(100, 160), gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    model.train()
    rng_loader_seeds = np.random.default_rng(seed=training_seed)
    for _ in (pbar := tqdm.trange(pretrain_epochs, desc="Pre-training", unit="epoch")):
        num_samples = 0
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        train_loader.seed(int(rng_loader_seeds.integers(0, 2**32, dtype=np.uint32, size=())))
        for batch_xs, batch_ys in tqdm.tqdm(
            train_loader,
            desc="Current epoch",
            unit="batch",
            leave=False,
            disable=not verbose,
        ):
            batch_xs = batch_xs.to(base.DEVICE)
            batch_ys = batch_ys.to(base.DEVICE)

            optimizer.zero_grad()
            batch_pred = model(batch_xs)
            batch_loss = loss(input=batch_pred, target=batch_ys)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * batch_xs.size(0)
            epoch_accuracy += (batch_pred.argmax(-1) == batch_ys).int().sum().item()
            num_samples += batch_xs.size(0)
        epoch_loss /= num_samples
        epoch_accuracy /= num_samples
        progress_dict = {
            "pretrain_epoch_loss": epoch_loss,
            "pretrain_epoch_accuracy": epoch_accuracy,
            "pretrain_lr": lr_scheduler.get_last_lr()[0],
        }

        pbar.set_postfix(progress_dict)

        lr_scheduler.step()
    mlflow.log_metrics(progress_dict, step=0)

    print("Evaluating pre-trained model")
    model.eval()
    pretrained_test_metrics, _ = _evaluate_model_test(model, data_generator)
    pretrained_test_metrics = {f"pretrain_{k}": v for k, v in pretrained_test_metrics.items()}
    print(f"Pre-train test accuracy: {pretrained_test_metrics['pretrain_test_accuracy']:.4f}")
    mlflow.log_metrics(pretrained_test_metrics, step=0)

    model.replace_dense(num_classes=10)
    return model, pretrained_test_metrics


def _build_model(num_classes: int) -> models.WideResNet:
    return models.WideResNet(
        in_channels=3,
        depth=16,
        widen_factor=4,
        num_classes=num_classes,
        use_group_norm=True,
        custom_init=True,
        swap_order=True,  # same network architecture as in DeepMind/TAN papers
        device=base.DEVICE,
        dtype=base.DTYPE,
    )


def _predict(
    model: torch.nn.Module,
    datapipe: torchdata.datapipes.iter.IterDataPipe,
    data_augmentation: bool,
    disable_tqdm: bool = False,
) -> torch.Tensor:
    # NB: Always returns data-augmentation dimension
    model.eval()
    datapipe = datapipe.map(torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True))
    normalize_transform = torchvision.transforms.v2.Normalize(
        mean=data.CIFAR10_MEAN,
        std=data.CIFAR10_STD,
    )
    if not data_augmentation:
        # Augmentations add normalization later
        datapipe = datapipe.map(normalize_transform)
    datapipe = datapipe.batch(base.EVAL_BATCH_SIZE, drop_last=False).collate()
    pred_logits = []
    with torch.no_grad():
        for batch_xs in tqdm.tqdm(datapipe, desc="Predicting", unit="batch", disable=disable_tqdm):
            if not data_augmentation:
                pred_logits.append(model(batch_xs.to(dtype=base.DTYPE, device=base.DEVICE)).cpu().unsqueeze(1))
            else:
                flip_augmentations = (False, True)
                shift_augmentations = (0, -4, 4)
                batch_xs_pad = torchvision.transforms.v2.functional.pad(
                    batch_xs,
                    padding=[4],
                )
                pred_logits_current = []
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
                pred_logits.append(torch.stack(pred_logits_current, dim=1))
    return torch.cat(pred_logits, dim=0)


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
    train_parser.add_argument("--num-epochs", type=int, default=200, help="Number of training epochs")
    train_parser.add_argument("--noise-multiplier", type=float, default=0.2, help="Gaussian noise multiplier")
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    train_parser.add_argument("--learning-rate", type=float, default=4.0, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=2048, help="Virtual batch size")
    train_parser.add_argument("--augmult-factor", type=int, default=8, help="Number of data augmentations per sample")
    train_parser.add_argument(
        "--pretraining-epochs", type=int, required=False, help="If set, pre-train for the given number of epochs"
    )

    # MIA
    attack_parser = subparsers.add_parser("attack")
    attack_parser.add_argument("--global-variance", action="store_true", help="Use global variance for LiRA")
    return parser.parse_args()


if __name__ == "__main__":
    main()
