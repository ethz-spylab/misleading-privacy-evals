import enum
import hashlib
import math
import pathlib
import typing

import numpy as np
import torch
import torch.utils.data
import torchdata.datapipes.iter
import torchdata.datapipes.map
import torchvision
import torchvision.transforms.v2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CanaryType(enum.Enum):
    CLEAN = "clean"
    LABEL_NOISE = "label_noise"  # for most
    BLANK_IMAGES = "blank_images"  # for DP-SGD
    OOD = "ood"  # for SSL
    SSL_WORST_CASE = "ssl"  # for SSL
    DUPLICATES_MISLABEL_HALF = "duplicates_mislabel_half"  # for SELENA
    DUPLICATES_MISLABEL_FULL = "duplicates_mislabel_full"  # for SELENA


class PoisonType(enum.Enum):
    RANDOM_IMAGES = "random_images"  # for RelaxLoss
    CANARY_DUPLICATES = "canary_duplicates"  # for SELENA (strong model)
    CANARY_DUPLICATES_NOISY = "canary_duplicates_noisy"
    NONCANARY_DUPLICATES_NOISY = "noncanary_duplicates_noisy"


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: typing.Optional[torch.Tensor] = None,
        indices: typing.Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._features = features
        self._labels = labels
        self._indices = (
            indices.to(dtype=torch.long) if indices is not None else torch.arange(len(features), dtype=torch.long)
        )
        if self._labels is not None:
            assert len(self._features) == len(self._labels)
        if self._indices is not None:
            assert 0 <= min(self._indices) and max(self._indices) < len(self._features)

    def __getitem__(self, index: int) -> torch.Tensor | typing.Tuple[torch.Tensor, torch.Tensor]:
        target_idx = self._indices[index]
        if self._labels is not None:
            return self._features[target_idx], self._labels[target_idx]
        else:
            return self._features[target_idx]

    def __len__(self):
        return len(self._indices)

    def as_unlabeled(self) -> "Dataset":
        return Dataset(features=self._features, labels=None, indices=self._indices)

    def build_datapipe(
        self,
        shuffle: bool = False,
        cycle: bool = False,
        add_sharding_filter: bool = False,
    ) -> torchdata.datapipes.iter.IterDataPipe:
        indices = tuple(range(len(self)))
        datapipe = torchdata.datapipes.map.SequenceWrapper(indices, deepcopy=False)

        if shuffle:
            datapipe = datapipe.shuffle()
        else:
            datapipe = datapipe.to_iter_datapipe()

        if cycle:
            datapipe = datapipe.cycle()

        if add_sharding_filter:
            datapipe = datapipe.sharding_filter()

        datapipe = datapipe.map(self.__getitem__)
        return datapipe

    def build_map_datapipe(self) -> torchdata.datapipes.map.MapDataPipe:
        indices = tuple(range(len(self)))
        datapipe = torchdata.datapipes.map.SequenceWrapper(indices, deepcopy=False)
        datapipe = datapipe.map(self.__getitem__)
        assert isinstance(datapipe, torchdata.datapipes.map.MapDataPipe)
        return datapipe

    @property
    def targets(self) -> torch.Tensor:
        if not self.is_labeled:
            raise ValueError("Dataset is not labeled")
        return self._labels[self._indices]

    @property
    def is_labeled(self) -> bool:
        return self._labels is not None

    def subset(
        self,
        indices: torch.Tensor,
        labels: typing.Optional[torch.Tensor] = None,
    ) -> "Dataset":
        assert torch.all(indices < len(self))

        selection_indices = self._indices[indices]
        new_features = self._features[selection_indices]
        if labels is not None:
            new_labels = labels
        else:
            new_labels = self._labels[selection_indices] if self._labels is not None else None
        assert len(new_labels) == len(indices)

        return Dataset(
            features=new_features,
            labels=new_labels,
        )


class SSLDataset(torch.utils.data.Dataset):
    """
    Only for SSL.
    Given the numpy array of the training data and the corresponding labels,
    return the data and the labels for the i-th shadow model.
    """

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            pos_1 = self.transform(data)
            pos_2 = self.transform(data)
        else:
            pos_1 = data
            pos_2 = data

        return pos_1, pos_2, target


class DatasetGenerator(object):
    _OOD_SAMPLES_MD5 = "47acc093c6ce95b2d453dff3a97f2500"
    _SSL_INDICES_MD5 = "dcc03147693d5d4c4f8f185fbd590b37"

    def __init__(
        self,
        num_shadow: int,
        num_canaries: int,
        canary_type: CanaryType,
        num_poison: int,
        poison_type: PoisonType,
        data_dir: pathlib.Path,
        seed: int,
        download: bool = False,
        fixed_halves: typing.Optional[bool] = None,  # this is just for validation of the setting
    ) -> None:
        # This always calculates all splits etc for consistency, and then only returns what's needed
        self._seed = seed
        self._num_shadow = num_shadow
        self._num_canaries = num_canaries
        self._canary_type = canary_type
        self._num_poison = num_poison
        self._poison_type = poison_type

        self._data_root = data_dir
        self._download = download

        # Load all features and labels into memory; requires < 900MB including replacing canaries
        self._clean_train_xs, self._clean_train_ys = self._load_cifar10(data_dir, train=True, download=download)
        self._test_xs, self._test_ys = self._load_cifar10(data_dir, train=False, download=download)

        # Only load OOD data if using OOD canaries, and SSL indices if using SSL canaries
        if self._canary_type == CanaryType.OOD:
            ood_samples_path = data_dir / "ood_imagenet_samples.pt"
            if not ood_samples_path.exists():
                raise FileNotFoundError(f"OOD canary image file {ood_samples_path} does not exist")
            ood_samples_signature = hashlib.md5(ood_samples_path.read_bytes()).hexdigest()
            if ood_samples_signature != self._OOD_SAMPLES_MD5:
                raise ValueError(
                    f"OOD canary image file {ood_samples_path} has wrong MD5 hash"
                    f" (expected {self._OOD_SAMPLES_MD5}, got {ood_samples_signature})"
                )

            self._ood_xs = torch.load(data_dir / "ood_imagenet_samples.pt")
            assert self._ood_xs.dtype == torch.uint8
            assert self._ood_xs.size()[1:] == (3, 32, 32)

            if self._ood_xs.size(0) < num_canaries:
                raise ValueError(f"Requested {num_canaries} OOD canaries, but only {self._ood_xs.size(0)} available")
        elif self._canary_type == CanaryType.SSL_WORST_CASE:
            ssl_indices_path = data_dir / "ssl_indices.pt"
            if not ssl_indices_path.exists():
                raise FileNotFoundError(f"SSL indices file {ssl_indices_path} does not exist")
            ssl_indices_signature = hashlib.md5(ssl_indices_path.read_bytes()).hexdigest()
            if ssl_indices_signature != self._SSL_INDICES_MD5:
                raise ValueError(
                    f"SSL indices file {ssl_indices_path} has wrong MD5 hash"
                    f" (expected {self._SSL_INDICES_MD5}, got {ssl_indices_path})"
                )

            self._ssl_indices = torch.load(data_dir / "ssl_indices.pt")
            assert self._ssl_indices.dtype == torch.int64
            assert self._ssl_indices.size() == (self.num_raw_training_samples,)

        assert len(self._clean_train_xs) == 50000
        assert len(self._test_xs) == 10000

        # Everything from here on depends on the seed
        # All indices are relative to the full raw training set
        # All index arrays (except label noise order) are stored sorted in increasing order
        rng = np.random.default_rng(seed=self._seed)

        num_raw_train_samples = len(self._clean_train_xs)
        num_classes = 10

        # 1) IN-OUT splits
        rng_splits_target, rng_splits_shadow, rng = rng.spawn(3)
        # Currently, we are not using any target models. However, keep rng for compatibility if we need them later.
        del rng_splits_target
        # This ensures that every sample is IN in exactly half of all shadow models if all samples were varied.
        # Calculate splits for all training samples, s.t. the membership is independent of the number of canaries
        # If the number of shadow models changes, then everything changes either way
        assert self._num_shadow % 2 == 0
        shadow_in_indices_t = np.argsort(
            rng_splits_shadow.uniform(size=(self._num_shadow, num_raw_train_samples)), axis=0
        )[: self._num_shadow // 2].T
        raw_shadow_in_indices = []
        for shadow_idx in range(self._num_shadow):
            raw_shadow_in_indices.append(
                torch.from_numpy(np.argwhere(np.any(shadow_in_indices_t == shadow_idx, axis=1)).flatten())
            )
        rng_splits_half, rng_splits_shadow = rng_splits_shadow.spawn(2)  # used later for fixed splits for validation
        del rng_splits_shadow

        # 2) Canary indices
        rng_canaries, rng = rng.spawn(2)
        self._canary_order = rng_canaries.permutation(num_raw_train_samples)
        # Replace canary order if SSL canary type
        # CAREFUL: This only works in isolation and without poisoning; else things might break down
        if self._canary_type == CanaryType.SSL_WORST_CASE:
            self._canary_order = self._ssl_indices
        del rng_canaries

        # Calculate proper IN indices depending on setting
        self._shadow_in_indices = []
        if fixed_halves is None:
            # Normal case; all non-canary samples are always IN
            canary_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
            canary_mask[self._canary_order[: self._num_canaries]] = True

            for shadow_idx in range(self._num_shadow):
                current_in_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                current_in_mask[raw_shadow_in_indices[shadow_idx]] = True
                current_in_mask[~canary_mask] = True
                self._shadow_in_indices.append(torch.argwhere(current_in_mask).flatten())
        else:
            # Special case to validate the setting
            # Always only use half of CIFAR10, but either vary by shadow model, or use a fixed half of non-canaries
            if not fixed_halves:
                # Raw shadow indices are already half of the full training data
                self._shadow_in_indices = raw_shadow_in_indices
            else:
                # Need to calculate a fixed half of non-canaries
                canary_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                canary_mask[self._canary_order[: self._num_canaries]] = True
                fixed_membership_full = torch.from_numpy(rng_splits_half.random(num_raw_train_samples) < 0.5)
                for shadow_idx in range(self._num_shadow):
                    current_in_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                    # IN: IN canaries and fixed non-canaries
                    current_in_mask[raw_shadow_in_indices[shadow_idx]] = True
                    current_in_mask[~canary_mask] = False
                    current_in_mask[(~canary_mask) & fixed_membership_full] = True
                    self._shadow_in_indices.append(torch.argwhere(current_in_mask).flatten())
        del rng_splits_half

        # 3) Canary transforms
        rng_canary_transforms, rng = rng.spawn(2)
        # 3.1) Noisy labels for all samples
        rng_noise, rng_canary_transforms = rng_canary_transforms.spawn(2)
        label_changes = torch.from_numpy(rng_noise.integers(num_classes - 1, size=num_raw_train_samples))
        self._noisy_labels = torch.where(label_changes < self._clean_train_ys, label_changes, label_changes + 1)
        del rng_noise
        # 3.2) Blank images w/ different colors and random labels
        rng_blank_images, rng_canary_transforms = rng_canary_transforms.spawn(2)
        # Store colors and labels for all samples, for consistency if # canaries changes
        # FIXME: Could happen that two canaries end up with same color, or that colors are close.
        #  But statistically very unlikely for only a few hundred canaries.
        self._blank_image_colors = torch.from_numpy(
            rng_blank_images.integers(
                0,
                256,
                size=(num_raw_train_samples, 3, 1, 1),
                dtype=np.uint8,
            )
        )
        self._blank_image_labels = torch.from_numpy(
            rng_blank_images.integers(0, num_classes, size=num_raw_train_samples)
        )
        del rng_blank_images
        del rng_canary_transforms

        # 4) Poisoning
        rng_poison, rng = rng.spawn(2)
        # 4.1) Random images
        rng_poison_images, rng_poison = rng_poison.spawn(2)
        # Just store a seed here, and generate poison samples iteratively and ad-hoc
        self._poison_random_images_seed = rng_poison_images.integers(0, 2**32, dtype=np.uint32)
        del rng_poison_images
        # 4.2) Duplicate canaries and flip labels of duplicates
        rng_poison_duplicates_labels, rng_poison = rng_poison.spawn(2)
        # Don't store noisy labels but offsets, to ensure every poison label is different
        self._poison_duplicate_label_changes = torch.from_numpy(
            rng_poison_duplicates_labels.integers(num_classes - 1, size=num_raw_train_samples)
        )
        del rng_poison_duplicates_labels
        del rng_poison

        del rng

    def _load_cifar10(
        self, data_dir: pathlib.Path, train: bool, download: bool
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        raw_dataset = torchvision.datasets.CIFAR10(
            str(data_dir),
            train=train,
            transform=torchvision.transforms.v2.Compose(
                [
                    torchvision.transforms.v2.ToImage(),
                    torchvision.transforms.v2.ToDtype(torch.uint8, scale=True),
                ]
            ),
            download=download,
        )
        xs = torch.empty((len(raw_dataset), 3, 32, 32), dtype=torch.uint8)
        ys = torch.empty((len(raw_dataset),), dtype=torch.long)
        for idx, (x, y) in enumerate(raw_dataset):
            xs[idx] = x
            ys[idx] = y
        return xs, ys

    def build_test_data(self) -> Dataset:
        return Dataset(
            self._test_xs,
            self._test_ys,
        )

    def build_full_train_data(
        self,
    ) -> Dataset:
        # NB: Full training data does NOT include poisons! This is for evaluation only!
        return self._build_train_data(
            indices=torch.arange(self.num_raw_training_samples),
            include_poison=False,
        )

    def build_train_data(
        self,
        shadow_model_idx: int,
    ) -> Dataset:
        return self._build_train_data(
            indices=self._shadow_in_indices[shadow_model_idx],
            include_poison=True,
        )

    def build_train_data_full_with_poison(
        self,
        shadow_model_idx: int,
    ) -> typing.Tuple[Dataset, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NB: Special case for SELENA, because there we require exact knowledge of canaries and poisons
        full_data = self._build_train_data(
            indices=torch.arange(self.num_raw_training_samples),
            include_poison=True,
        )

        canary_mask = torch.zeros((len(full_data),), dtype=torch.bool)
        canary_mask[self.get_canary_indices()] = True

        membership_mask = torch.zeros_like(canary_mask)
        membership_mask[self._shadow_in_indices[shadow_model_idx]] = True

        poison_mask = torch.zeros_like(canary_mask)
        if self._num_poison > 0:
            # Poison indices are always appended at the end
            poison_mask[-self._num_poison :] = True
            membership_mask[-self._num_poison :] = True

        return full_data, membership_mask, canary_mask, poison_mask

    def build_attack_data_with_poison(
        self,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NB: Special case for SELENA (Split-AI only), because there we require exact knowledge of canaries and poisons

        full_data = self._build_train_data(
            indices=torch.arange(self.num_raw_training_samples),
            include_poison=True,
        )
        attack_ys = full_data.targets

        shadow_membership_mask = torch.zeros((len(attack_ys), self._num_shadow), dtype=torch.bool)
        for shadow_model_idx in range(self._num_shadow):
            shadow_membership_mask[self._shadow_in_indices[shadow_model_idx], shadow_model_idx] = True

        canary_mask = torch.zeros((len(attack_ys),), dtype=torch.bool)
        if self._canary_type != CanaryType.DUPLICATES_MISLABEL_HALF:
            canary_mask[self.get_canary_indices()] = True
        else:
            canary_indices_clean = self.get_canary_indices()[: self._num_canaries // 2]
            canary_indices_mislabel = self.get_canary_indices()[-self._num_canaries // 2 :]
            # Want to audit only MISLABELED part, but need to query clean samples
            canary_mask[canary_indices_clean] = True

            # Hence, need to swap targets and membership values
            attack_ys_new = torch.clone(attack_ys)
            attack_ys_new[canary_indices_mislabel] = attack_ys[canary_indices_clean]
            attack_ys_new[canary_indices_clean] = attack_ys[canary_indices_mislabel]
            attack_ys = attack_ys_new

            shadow_membership_mask_new = torch.clone(shadow_membership_mask)
            shadow_membership_mask_new[canary_indices_mislabel] = shadow_membership_mask[canary_indices_clean]
            shadow_membership_mask_new[canary_indices_clean] = shadow_membership_mask[canary_indices_mislabel]
            shadow_membership_mask = shadow_membership_mask_new

        poison_mask = torch.zeros_like(canary_mask)
        if self._num_poison > 0:
            # Poison indices are always appended at the end
            poison_mask[-self._num_poison :] = True
            shadow_membership_mask[-self._num_poison :, :] = True

        return attack_ys, shadow_membership_mask, canary_mask, poison_mask

    def build_attack_data(
        self,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, attack_ys = self._build_train_canary_data()

        canary_indices = torch.from_numpy(self.get_canary_indices())

        # Special case for SELENA: only the second half (mislabeled) are canaries for attack
        if self._canary_type == CanaryType.DUPLICATES_MISLABEL_HALF:
            assert len(canary_indices) % 2 == 0, "Need even number of canaries to use half"
            canary_indices = canary_indices[-self._num_canaries // 2 :]

        # As usual, this mask does not include poisons
        shadow_membership_mask = torch.zeros((self.num_raw_training_samples, self._num_shadow), dtype=torch.bool)
        for shadow_model_idx in range(self._num_shadow):
            shadow_membership_mask[:, shadow_model_idx] = self.build_in_mask(shadow_model_idx)

        return attack_ys, shadow_membership_mask, canary_indices

    def _build_train_data(
        self,
        indices: torch.Tensor,
        include_poison: bool,
    ) -> Dataset:
        # First, apply canaries to training data
        train_xs, train_ys = self._build_train_canary_data()
        if include_poison and self._num_poison > 0:
            if self._poison_type == PoisonType.RANDOM_IMAGES:
                poison_xs, poison_ys = self._build_poison_random_images()
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
                assert poison_indices[-1] == len(train_xs) - 1
            elif self._poison_type == PoisonType.CANARY_DUPLICATES:
                assert self._num_poison == self._num_canaries, "Currently only support duplicating every canary"
                poison_indices = torch.from_numpy(self.get_canary_indices())
            elif self._poison_type == PoisonType.CANARY_DUPLICATES_NOISY:
                assert self._num_poison == self._num_canaries, "Currently only support duplicating every canary"
                # train_xs/_ys are independent of IN/OUT; hence get all canaries as desired
                canary_indices = self.get_canary_indices()
                poison_xs = train_xs[canary_indices]
                canary_ys = train_ys[canary_indices]
                label_changes = self._poison_duplicate_label_changes[canary_indices]
                poison_ys = torch.where(label_changes < canary_ys, label_changes, label_changes + 1)
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
                assert poison_indices[-1] == len(train_xs) - 1
            elif self._poison_type == PoisonType.NONCANARY_DUPLICATES_NOISY:
                # Use non-canaries from back to front, s.t. increasing # canaries still yields the same poisons
                assert self._num_poison + self._num_canaries <= train_xs.size(0)
                duplicate_indices = self.get_non_canary_indices()[-self._num_poison :]
                poison_xs = train_xs[duplicate_indices]
                duplicate_ys = train_ys[duplicate_indices]
                label_changes = self._poison_duplicate_label_changes[duplicate_indices]
                poison_ys = torch.where(label_changes < duplicate_ys, label_changes, label_changes + 1)
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
                assert poison_indices[-1] == len(train_xs) - 1
            else:
                assert False, f"Unknown poison type {self._poison_type}"

            # Always append poison indices at the end
            assert poison_indices.size() == (self._num_poison,)
            indices = torch.cat((indices, poison_indices))

        return Dataset(
            train_xs,
            train_ys,
            indices=indices,
        )

    def _build_train_canary_data(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # This always operates on all training samples, independent of IN vs. OUT
        if self._canary_type == CanaryType.CLEAN:
            return self._clean_train_xs, self._clean_train_ys
        elif self._canary_type == CanaryType.LABEL_NOISE:
            noisy_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            noisy_targets[canary_indices] = self._noisy_labels[canary_indices]
            return self._clean_train_xs, noisy_targets
        elif self._canary_type == CanaryType.BLANK_IMAGES:
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            canary_features[canary_indices] = self._blank_image_colors[canary_indices]
            canary_targets[canary_indices] = self._blank_image_labels[canary_indices]
            return canary_features, canary_targets
        elif self._canary_type == CanaryType.OOD:
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            assert canary_indices.shape[0] <= self._ood_xs.size(0)
            # OOD canaries replace original ID canaries
            # Use the labels of the original canaries to be random but keep things balanced
            canary_features[canary_indices] = self._ood_xs[: canary_indices.shape[0]]
            return canary_features, canary_targets
        elif self._canary_type == CanaryType.SSL_WORST_CASE:
            # Overwrites canary order with SSL worst-case order, hence just return clean data
            return self._clean_train_xs, self._clean_train_ys
        elif self._canary_type in (CanaryType.DUPLICATES_MISLABEL_HALF, CanaryType.DUPLICATES_MISLABEL_FULL):
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            # Duplicate first half of canaries
            assert len(canary_indices) % 2 == 0, "Need even number of canaries to use half"
            canary_indices_original = canary_indices[: self._num_canaries // 2]
            canary_indices_duplicate = canary_indices[self._num_canaries // 2 :]
            assert len(canary_indices_original) == len(canary_indices_duplicate)
            canary_features[canary_indices_duplicate] = canary_features[canary_indices_original]
            # Always mislabel half of canaries (need to use original noisy labels)
            canary_targets[canary_indices_duplicate] = self._noisy_labels[canary_indices_original]
            if self._canary_type == CanaryType.DUPLICATES_MISLABEL_FULL:
                # Also use same wrong labels for other half
                canary_targets[canary_indices_original] = canary_targets[canary_indices_duplicate]
            return canary_features, canary_targets
        else:
            assert False, f"Unknown canary type {self._canary_type}"

    def _build_poison_random_images(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self._poison_random_images_seed)

        # Add 10 instances of each poisoned sample => round up # poisons to a multiple of 10
        num_classes = 10
        num_poison_total = math.ceil(self._num_poison / num_classes) * num_classes
        num_random_images = num_poison_total // num_classes
        poison_targets = torch.tile(torch.arange(num_classes, dtype=torch.long), (num_random_images,))
        poison_features = torch.zeros((num_poison_total, 3, 32, 32), dtype=torch.uint8)
        for idx in range(num_random_images):
            poison_features[idx * num_classes : (idx + 1) * num_classes] = torch.from_numpy(
                rng.integers(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
            )
        return poison_features, poison_targets

    def build_in_mask(self, shadow_model_idx: int) -> torch.Tensor:
        # NB: This does NOT include poisons!
        result = torch.zeros(self.num_raw_training_samples, dtype=torch.bool)
        result[self._shadow_in_indices[shadow_model_idx]] = True
        return result

    def get_canary_indices(
        self,
    ) -> np.ndarray:
        # Poisons are always appended to the data, hence the indices here are always correct
        return self._canary_order[: self._num_canaries]

    def get_non_canary_indices(self) -> np.ndarray:
        return self._canary_order[self._num_canaries :]

    @property
    def num_raw_training_samples(self) -> int:
        return len(self._clean_train_xs)

    @property
    def num_shadow(self) -> int:
        return self._num_shadow

    @property
    def num_canaries(self) -> int:
        return self._num_canaries

    @property
    def canary_type(self) -> CanaryType:
        return self._canary_type

    @property
    def num_poison(self) -> int:
        return self._num_poison

    @property
    def poison_type(self) -> PoisonType:
        return self._poison_type

    def load_cifar100(self) -> torchvision.datasets.CIFAR100:
        return torchvision.datasets.CIFAR100(
            root=str(self._data_root),
            train=True,
            download=self._download,
            transform=torchvision.transforms.v2.Compose(
                [
                    torchvision.transforms.v2.ToImage(),
                    torchvision.transforms.v2.ToDtype(torch.uint8, scale=True),
                ]
            ),
        )

    def build_train_ssl_data(
        self,
        shadow_model_idx: typing.Optional[int],
        transform: torchvision.transforms.v2.Compose,
    ) -> SSLDataset:
        indices = self._shadow_in_indices[shadow_model_idx]
        # First, apply canaries to training data
        train_xs, train_ys = self._build_train_canary_data()
        return SSLDataset(
            train_xs[indices],
            train_ys[indices],
            transform,
        )

    def build_full_train_ssl_data(
        self,
        transform: torchvision.transforms.v2.Compose = None,
    ) -> SSLDataset:
        train_xs, train_ys = self._build_train_canary_data()

        return SSLDataset(
            train_xs,
            train_ys,
            transform,
        )
