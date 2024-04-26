import argparse
import json
import os
import pathlib
import typing

import PIL.Image
import dotenv
import filelock
import mlflow
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchdata.datapipes.map
import torchvision
import torchvision.transforms.v2
import tqdm

import attack_util
import base
import data
import models


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, tensor, mean, std, reverse=False):
        if reverse:
            _mean = [-m / s for m, s in zip(mean, std)]
            _std = [1 / s for s in std]
        else:
            _mean = mean
            _std = std

        _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
        _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
        tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
        return tensor

    def __call__(self, x, reverse=False):
        return self.normalize(x, self.mean, self.std, reverse=reverse)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction="batchmean"):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def kldiv(self, logits, targets, T=1.0, reduction="batchmean"):
        q = F.log_softmax(logits / T, dim=1)
        p = F.softmax(targets / T, dim=1)
        return F.kl_div(q, p, reduction=reduction) * (T * T)

    def forward(self, logits, targets):
        return self.kldiv(logits, targets, T=self.T, reduction=self.reduction)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()


class UnlabeledNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)

        def read_all_npy_files(root):
            files = []
            for dirpath, dirnames, filenames in os.walk(root):
                for f in filenames:
                    if f.endswith(".npy"):
                        files.append(os.path.join(dirpath, f))
            return len(files) * 256  # synthetic data size

        self.size = read_all_npy_files(self.root)
        print(f"synthetic data size: {self.size}")
        self.to_pil = torchvision.transforms.v2.ToPILImage()
        self.transform = transform

    def __getitem__(self, idx):
        # ! use lazy loading
        img = np.load(os.path.join(self.root, "%d.npy" % (idx // 256)))[idx % 256]
        img = self.to_pil(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.size


class SynDataset(torchvision.datasets.CIFAR10):
    def __init__(self, numpy_data, transform=None):
        super(SynDataset, self).__init__(root=data_dir, train=False, download=True, transform=transform)
        self.data = numpy_data
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        img = PIL.Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)


class DeepInversionHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(
                module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2
            ) + torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (
                self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data,
            )

    def remove(self):
        self.hook.remove()


class DataPool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        imgs = imgs.transpose(0, 2, 3, 1)
        np.save(os.path.join(self.root, "%d.npy" % (self._idx)), imgs)
        self._idx += 1

    def get_dataset(self, transform=None):
        return UnlabeledNumpyDataset(self.root, transform=transform)


class Synthesizer:
    def __init__(
        self,
        teacher,
        student_net,
        generator,
        nz,
        num_classes,
        img_size,
        iterations=100,
        lr_g=0.1,
        synthesis_batch_size=128,
        sample_batch_size=128,
        adv=0.0,
        bn=1,
        oh=1,
        save_dir="run/fast",
        transform=None,
        normalizer=None,
        lr_z=0.01,
        warmup=10,
        bn_mmt=0,
        rdm_g=0,
    ):
        super(Synthesizer, self).__init__()

        self.teacher = teacher
        self.student_net = student_net
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.rdm_g = rdm_g
        self.num_classes = num_classes

        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.data_pool = DataPool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.memory_data = []

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

        self.aug = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                torchvision.transforms.v2.RandomHorizontalFlip(),
                torchvision.transforms.v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ]
        )

    def synthesize(self, targets=None):
        self.ep += 1
        self.student_net.eval()
        self.teacher.eval()
        best_cost = 1e6

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda().requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.cuda()

        # ! randomize the generator
        if self.rdm_g > 0:
            self.generator = Generator(nz=256, ngf=64, img_size=32, nc=3).cuda().train()

        optimizer = torch.optim.Adam(
            [{"params": self.generator.parameters()}, {"params": [z], "lr": self.lr_z}],
            lr=self.lr_g,
            betas=[0.5, 0.999],
        )

        for it in range(self.iterations):
            inputs = self.generator(z)
            inputs_aug = self.aug(inputs)  # crop and normalize

            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(inputs_aug)
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.cuda()

            # ! only use the bn loss
            loss = sum([h.r_feature for h in self.hooks])
            tqdm.tqdm.write(f"it: {it}, loss: {loss.item():.2f}")
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        self.student_net.train()
        tmp_data = (best_inputs.detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        tmp_data = tmp_data.transpose(0, 2, 3, 1)
        self.memory_data.append(tmp_data)

    def get_syn_data_loader(self):
        syn_data = np.concatenate(np.array(self.memory_data), 0)
        syn_dataset = SynDataset(syn_data, transform=self.transform)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=self.sample_batch_size, shuffle=True, num_workers=4
        )
        return syn_data_loader


def main():
    dotenv.load_dotenv()
    args = parse_args()
    global data_dir
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
    num_epochs = 240

    print(f"Training shadow model {shadow_model_idx}")
    print(
        f"{data_generator.num_canaries} canaries ({data_generator.canary_type.value}), "
        f"{data_generator.num_poison} poisons ({data_generator.poison_type.value})"
    )

    output_dir = directory_manager.get_training_output_dir(shadow_model_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = directory_manager.get_training_log_dir(shadow_model_idx)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_img_dir = directory_manager.get_training_img_dir(shadow_model_idx)
    save_img_dir.mkdir(parents=True, exist_ok=True)

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
        # Load teacher model
        # NB: Teacher is trained using undefended.py
        teacher_net = models.WideResNet(
            in_channels=3,
            depth=16,
            widen_factor=4,
            num_classes=10,
            use_group_norm=False,
            device=base.DEVICE,
            dtype=base.DTYPE,
        )
        weight = torch.load(
            directory_manager.get_teacher_model_path(args.teacher_dir, shadow_model_idx=shadow_model_idx)
        )
        teacher_net.load_state_dict(weight)
        teacher_net.eval()
        mlflow.log_params(
            {
                "shadow_model_idx": shadow_model_idx,
                "num_canaries": data_generator.num_canaries,
                "canary_type": data_generator.canary_type.value,
                "num_poison": data_generator.num_poison,
                "poison_type": data_generator.poison_type.value,
                "training_seed": training_seed,
            }
        )
        if args.eval_only:
            current_model = models.WideResNet(
                in_channels=3,
                depth=16,
                widen_factor=4,
                num_classes=10,
                use_group_norm=False,
                device=base.DEVICE,
                dtype=base.DTYPE,
            )
            current_model.load_state_dict(torch.load(output_dir / "model.pt"))
            print("Loaded model, and evaluating")
            test_metrics, test_pred = _evaluate_model_test(current_model, data_generator)
            metrics = dict()
            metrics.update(test_metrics)
            print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        else:
            current_model = _train_model(
                train_data,
                training_seed=training_seed,
                num_epochs=num_epochs,
                verbose=verbose,
                teacher_net=teacher_net,
                save_dir=save_img_dir / f"syn_{shadow_model_idx}",
                output_dir=output_dir,
            )
        current_model.eval()
        torch.save(current_model.state_dict(), output_dir / "model.pt")
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
    output_dir = directory_manager.get_attack_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    # import pdb; pdb.set_trace()
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
        print(f"TPR at FPR {target_fpr * 100}%: {tpr[fpr <= target_fpr][-1] * 100:.4f}%")

    # Calculate attack accuracy
    prediction_threshold = torch.median(attack_scores).item()
    pred_membership = attack_scores > prediction_threshold  # median returns lower of two values => strict ineq.
    balanced_accuracy = torch.mean((pred_membership == attack_membership).float()).item()
    print(f"Attack accuracy: {balanced_accuracy:.4f}")


def _train_student(syn_data_loader, student, teacher, criterion, optimizer):
    student.train()
    teacher.eval()
    # ! do the distillation for one iteration
    loss = 0
    for iter in range(5):
        for _, images in enumerate(syn_data_loader):
            images = images.cuda()

            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()

            loss_s.backward()
            loss += loss_s.item()
            optimizer.step()
    print(
        f"synthetic data size: {len(syn_data_loader.dataset)}, training loss: {loss / len(syn_data_loader.dataset) / 5}"
    )


def _train_model(
    train_data: data.Dataset,
    training_seed: int,
    num_epochs: int,
    verbose: bool = False,
    teacher_net: torch.nn.Module = None,
    save_dir: pathlib.Path = None,
    output_dir: pathlib.Path = None,
) -> torch.nn.Module:
    # More or less HPs from SELENA
    lr = 0.1  # for cnn
    momentum = 0.9
    weight_decay = 5e-4

    student_net = models.WideResNet(
        in_channels=3,
        depth=16,
        widen_factor=4,
        num_classes=10,
        use_group_norm=False,
        device=base.DEVICE,
        dtype=base.DTYPE,
    )
    # ! get synthesizer
    nz = 256
    transform_syn = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomCrop(32, padding=4),
            torchvision.transforms.v2.RandomHorizontalFlip(),
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    normalizer = Normalizer(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    generator = Generator(nz=nz, ngf=64, img_size=32, nc=3)
    generator = generator.cuda()
    synthesizer = Synthesizer(
        teacher_net,
        student_net,
        generator,
        nz=nz,
        num_classes=10,
        img_size=(3, 32, 32),
        save_dir=save_dir,
        transform=transform_syn,
        normalizer=normalizer,
        synthesis_batch_size=256,
        sample_batch_size=256,
        iterations=2,
        warmup=20,
        lr_g=5e-3,
        lr_z=0.015,
        adv=0,
        bn=10,
        oh=0,
        bn_mmt=0.9,
        rdm_g=0,
    )

    optimizer = torch.optim.SGD(student_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)
    criterion = KLDiv(T=20)
    # read cifar10 testdata
    global transform_test
    transform_test = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    best_acc = -1

    def test_accuracy(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for features, target in test_loader:
                features, target = features.cuda(), target.cuda()
                output = model(features)
                test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
                pred = torch.max(output, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        return acc, test_loss

    for epoch in tqdm.tqdm(range(num_epochs)):
        student_net.train()
        synthesizer.synthesize()  # g_steps

        if epoch >= 20:
            student_net.train()
            syn_loader = synthesizer.get_syn_data_loader()
            _train_student(syn_loader, student_net, teacher_net, criterion, optimizer)  # kd_steps
            lr_scheduler.step()
            student_net.eval()
            test_acc, test_loss = test_accuracy(student_net, val_loader)
            if test_acc > best_acc:
                torch.save(student_net.state_dict(), output_dir / "model.pt")
                best_acc = test_acc

            print(f"student_net, epoch {epoch}, acc: {test_acc:.2f}, best: {best_acc:.2f}, test_loss: {test_loss:.2f}")

    student_net.eval()
    return student_net


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

    def get_teacher_model_path(self, teach_dir: str, shadow_model_idx: typing.Optional[int]) -> pathlib.Path:
        teach_dir = pathlib.Path(teach_dir)
        return teach_dir / str(shadow_model_idx) / "model.pt"

    def get_training_img_dir(self, shadow_model_idx: typing.Optional[int]) -> pathlib.Path:
        return self._experiment_dir / "img"


def parse_args() -> argparse.Namespace:
    default_data_dir = pathlib.Path(os.environ.get("DATA_ROOT", "data"))
    default_base_experiment_dir = pathlib.Path(os.environ.get("EXPERIMENT_DIR", ""))

    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--data-dir", type=pathlib.Path, default=default_data_dir, help="Dataset root directory")
    parser.add_argument(
        "--experiment-dir", default=default_base_experiment_dir, type=pathlib.Path, help="Experiment directory"
    )
    parser.add_argument("--teacher_dir", type=str, default="", help="Teacher model directory")

    parser.add_argument("--experiment", type=str, default="dev", help="Experiment name")
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional run suffix to distinguish multiple runs in the same experiment",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

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

    # MIA
    attack_parser = subparsers.add_parser("attack")  # noqa: F841

    return parser.parse_args()


if __name__ == "__main__":
    main()
