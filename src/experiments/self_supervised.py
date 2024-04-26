import argparse
import json
import math
import os
import pathlib
import typing
import warnings

import dotenv
import filelock
import mlflow
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms.v2
import tqdm

import attack_util
import base
import data
from undefended import _evaluate_model_test, _predict

warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, method: str, net: nn.Module, num_class: int = 10):
        super(Net, self).__init__()
        if method == "simclr":
            self.encoder = net.encoder
        elif method == "moco":
            self.encoder = torch.nn.Sequential(*list(net.encoder_q.net.children())[:-2])
        else:
            raise NotImplementedError
        self.fc = nn.Linear(512, num_class, bias=True)

    def load_backbone(self, method: str, pretrained_path: pathlib.Path):
        print(f"Loading pretrained model from {pretrained_path}")
        # load the pretrained backbone
        weight = torch.load(pretrained_path, map_location="cpu")
        count = 0
        if method == "simclr":
            # ! for simclr, just load the encoder
            for key_ori, value in self.encoder.state_dict().items():
                key = "encoder." + key_ori
                if key in weight:
                    if value.shape != weight[key].shape:
                        raise ValueError(
                            f"Shape mismatch: {key} has shape {value.shape} but {pretrained_path} has shape {weight[key].shape}"
                        )
                    else:
                        count += 1
                        value.copy_(weight[key])
                else:
                    raise ValueError(f"Missing key: {key} in {pretrained_path}")

        elif method == "moco":
            # for moco, only load the encoder_q
            # ! do not load the classifier and flatten layer
            for key_ori, value in self.encoder.state_dict().items():
                key = "net." + key_ori
                if key in weight:
                    if value.shape != weight[key].shape:
                        raise ValueError(
                            f"Shape mismatch: {key} has shape {value.shape} but {pretrained_path} has shape {weight[key].shape}"
                        )
                    else:
                        value.copy_(weight[key])
                else:
                    raise ValueError(f"Missing key: {key} in {pretrained_path}")
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = self.encoder(x)  # [512, 512, 1, 1]
        feature = torch.flatten(tmp, start_dim=1)
        out = self.fc(feature)
        return out


def test_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> typing.Tuple[float, float]:
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


def setup_seed(seed: int):
    import torch.backends.cudnn as cudnn
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


# test using a knn monitor
def knn_test(net: nn.Module, train_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader):
    net.eval()
    classes = 10
    total_top1, total_num, feature_bank = 0.0, 0, []
    feature_labels = []

    with torch.no_grad():
        for batch_xs_1, batch_xs_2, labels in train_loader:
            im_1 = batch_xs_1.to(base.DEVICE)
            feature = net(im_1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.extend(labels)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(np.array(feature_labels), device=feature_bank.device)

        def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1
            )

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            return pred_labels

        for batch_xs_1, labels in test_data_loader:
            features, labels = batch_xs_1.to(base.DEVICE), labels.to(base.DEVICE)
            feature = net(features)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

            total_num += features.size(0)
            total_top1 += (pred_labels[:, 0] == labels).float().sum().item()

        print("KNN TRAIN Acc@1:{:.2f}%".format(total_top1 / total_num * 100))

    return total_top1 / total_num * 100


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim: int = 128, arch: str = None, bn_splits: int = 16):
        super(ModelBase, self).__init__()

        norm_layer = nn.BatchNorm2d
        resnet_arch = getattr(torchvision.models.resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net[:-2](x)
        output = output.view(output.size(0), -1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # note: not normalized here
        return x


class MoCo(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        K: int = 4096,
        m: float = 0.99,
        T: float = 0.1,
        arch: str = "resnet18",
        bn_splits: int = 8,
        symmetric: bool = True,
    ):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x: torch.Tensor):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def get_head_embedding(self, im_q, im_k):
        q = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)
        k = self.encoder_q(im_k)  # keys: NxC
        # k = nn.functional.normalize(k, dim=1)
        return q, k

    def get_backbone_embedding(self, im_q, im_k):
        q = self.encoder_q.get_feature(im_q)
        # q = nn.functional.normalize(q, dim=1)
        k = self.encoder_q.get_feature(im_k)
        # k = nn.functional.normalize(k, dim=1)

        return q, k

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss


class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, arch="resnet18"):
        super(SimCLR, self).__init__()

        self.f = []
        backbone = torchvision.models.resnet18()
        for name, module in backbone.named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.encoder = nn.Sequential(*self.f)
        # projection head
        out_dim = 512 if arch == "resnet18" else 2048
        self.proj_head = nn.Sequential(
            nn.Linear(out_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.proj_head(feature)
        return out

    def get_backbone_embedding(self, x1, x2):
        x1 = self.encoder(x1)
        feature1 = torch.flatten(x1, start_dim=1)
        # feature1 = F.normalize(feature1, dim=1)

        x2 = self.encoder(x2)
        feature2 = torch.flatten(x2, start_dim=1)
        # feature2 = F.normalize(feature2, dim=1)

        return feature1, feature2

    def get_head_embedding(self, x1, x2):
        res1 = self.forward(x1)
        # res1 = F.normalize(res1, dim=1)
        res2 = self.forward(x2)
        # res2 = F.normalize(res2, dim=1)
        return res1, res2

    def loss_func(self, img1, img2, temp=0.5):
        z1 = F.normalize(img1, dim=1)
        z2 = F.normalize(img2, dim=1)

        N, Z = z1.shape
        device = z1.device

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.T)

        # create positive matches
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

        # print(positives)

        # get the values of every pair that's a mismatch
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temp
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction="sum")

        return loss / (2 * N)

    def get_loss(self, img1, img2):
        feature_1, feature_2 = self.forward(img1), self.forward(img2)
        loss = self.loss_func(feature_1, feature_2)
        return loss


def main():
    dotenv.load_dotenv()
    args = parse_args()
    global data_dir, method
    method = args.method
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
    # ! only finetune=1 for ["similarity", "confidence"], finetune=0 for ["head", "backbone"]
    if args.finetune == 1:
        assert args.score_type in ["similarity", "confidence"]
    else:
        assert args.score_type in ["head", "backbone"]

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
        global shadow_model_idx
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


def _train_ssl(
    train_data: data.SSLDataset,
    memory_data: data.SSLDataset,
    test_loader: torch.utils.data.DataLoader,
    training_seed: int,
    num_epochs: int,
    verbose: bool = False,
    output_dir: pathlib.Path = None,
    args: argparse.Namespace = None,
) -> torch.nn.Module:
    batch_size = 512
    momentum = 0.9
    weight_decay = 5e-4
    learning_rate = 0.06  # 0.1

    if args.method == "simclr":
        model = SimCLR(feature_dim=128).cuda()
    elif args.method == "moco":
        model = MoCo(
            dim=128,
            K=4096,  # queue size, default 4096
            m=0.99,  # momentum
            T=0.1,  # temperature
            bn_splits=8,
            symmetric=True,
        ).cuda()
    else:
        assert False, f"Unknown method {args.method}"

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    memory_loader = torch.utils.data.DataLoader(dataset=memory_data, shuffle=False, batch_size=512, num_workers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    model.train()

    # lr scheduler for training
    def adjust_learning_rate(optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = learning_rate
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / num_epochs))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # train for one epoch
    def train(net, data_loader, train_optimizer, epoch, args):
        net.train()
        optimizer = train_optimizer
        adjust_learning_rate(optimizer, epoch, args)

        total_loss, total_num = 0.0, 0
        for im_1, im_2, _ in data_loader:
            im_1, im_2 = im_1.cuda(), im_2.cuda()
            # forward pass
            if args.method == "simclr":
                loss = net.get_loss(im_1, im_2)
            # for moco, do the forward pass
            elif args.method == "moco":
                loss = net(im_1, im_2)
            else:
                raise NotImplementedError()

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size

        return total_loss / total_num

    for epoch in (pbar := tqdm.trange(num_epochs, desc="Training", unit="epoch")):
        train(model, train_loader, optimizer, epoch, args)
        backbone = model.encoder_q if args.method == "moco" else model
        if epoch % 10 == 0:
            knn_test(backbone, memory_loader, test_loader)

        torch.save(backbone.state_dict(), output_dir / "model.pt")

    return model


def _finetune(
    train_data: data.SSLDataset,
    test_loader: torch.utils.data.DataLoader,
    training_seed: int,
    num_epochs: int,
    verbose: bool = False,
    output_dir: pathlib.Path = None,
    args: argparse.Namespace = None,
) -> torch.nn.Module:
    batch_size = 256
    if args.method == "simclr":
        arch = SimCLR(feature_dim=128).cuda()
    elif args.method == "moco":
        arch = MoCo(
            dim=128,
            K=4096,  # queue size
            m=0.99,  # momentum
            T=0.1,  # temperature
            bn_splits=8,
            symmetric=True,
        ).cuda()
    else:
        raise NotImplementedError()

    model = Net(args.method, arch)
    model.load_backbone(args.method, pretrained_path=output_dir.parent / "backbone" / "model.pt")
    model = model.cuda()
    for param in model.encoder.parameters():
        param.requires_grad = False

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.5, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    best_acc = 0

    model.train()
    for epoch in tqdm.trange(num_epochs, desc="Training", unit="epoch"):
        model.train()
        for batch_idx, (inputs, _, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        acc, _ = test_accuracy(model, test_loader)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}, test accuracy: {acc:.2f}%, best accuracy: {best_acc:.2f}%")
        torch.save(model.fc.state_dict(), output_dir / "model.pt")

    return model


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
    num_epochs = 800 if args.finetune == 0 else 100
    print(f"Training shadow model {shadow_model_idx}")
    print(
        f"{data_generator.num_canaries} canaries ({data_generator.canary_type.value}), "
        f"{data_generator.num_poison} poisons ({data_generator.poison_type.value})"
    )

    output_dir = directory_manager.get_training_output_dir(shadow_model_idx, finetune=args.finetune)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = directory_manager.get_training_log_dir(shadow_model_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    ssl_transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomResizedCrop(32, antialias=True),
            torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.v2.RandomGrayscale(p=0.2),
            torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
            torchvision.transforms.v2.Normalize(
                mean=data.CIFAR10_MEAN,
                std=data.CIFAR10_STD,
            ),
        ]
    )
    test_transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
            torchvision.transforms.v2.Normalize(
                mean=data.CIFAR10_MEAN,
                std=data.CIFAR10_STD,
            ),
        ]
    )

    train_data = data_generator.build_train_ssl_data(shadow_model_idx=shadow_model_idx, transform=ssl_transform)
    memory_data = data_generator.build_full_train_ssl_data(transform=test_transform)
    memory_loader = torch.utils.data.DataLoader(memory_data, batch_size=256, shuffle=False, num_workers=4)

    print(f"Dataset size:{len(train_data)}, shadow_model_idx:{shadow_model_idx}")

    test_data = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data.CIFAR10_MEAN, std=data.CIFAR10_STD),
            ]
        ),
        download=False,
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

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
            }
        )
        # 1) train the encoder
        if args.finetune == 0:
            # 1.1) load the encoder, and do evaluation to get features
            if args.eval_only:
                if args.method == "simclr":
                    current_model = SimCLR(feature_dim=128).cuda()
                elif args.method == "moco":
                    current_model = MoCo(
                        dim=128,
                        K=4096,  # queue size
                        m=0.99,  # momentum
                        T=0.1,  # temperature
                        bn_splits=8,
                        symmetric=True,
                    ).cuda()
                else:
                    raise NotImplementedError()
                weight = torch.load(output_dir / "model.pt")

                if args.method == "simclr":
                    current_model.load_state_dict(weight)
                elif args.method == "moco":
                    current_model.encoder_q.load_state_dict(weight)
                else:
                    assert False, f"Unknown method {args.method}"

                print("Backbone loaded")

            else:
                current_model = _train_ssl(
                    train_data=train_data,
                    memory_data=memory_data,
                    test_loader=test_loader,
                    training_seed=training_seed,
                    num_epochs=num_epochs,
                    verbose=verbose,
                    output_dir=output_dir,
                    args=args,
                )
            current_model.eval()

            print("Get features from encoder on full training data")

            full_train_data = data_generator.build_full_train_ssl_data(transform=None)
            train_pred_full = get_backbone_scores(
                current_model,
                full_train_data,
                test_loader=test_loader,
                memory_loader=memory_loader,
                data_augmentation=True,
                score_type=args.score_type,
            )
            np.save(output_dir / f"outputs_{args.score_type}.npy", train_pred_full)
            print("Features saved")

            # 2) train the fc layer
        else:
            if args.eval_only:
                if args.method == "simclr":
                    arch = SimCLR(feature_dim=128).cuda()
                elif args.method == "moco":
                    arch = MoCo(
                        dim=128,
                        K=4096,  # queue size
                        m=0.99,  # momentum
                        T=0.1,  # temperature
                        bn_splits=8,
                        symmetric=True,
                    ).cuda()
                else:
                    raise NotImplementedError
                current_model = Net(args.method, arch)

                backbone_path = output_dir.parent / "backbone" / "model.pt"
                fc_weight = torch.load(output_dir / "model.pt")
                current_model.load_backbone(args.method, backbone_path)
                current_model.fc.load_state_dict(fc_weight)
                current_model = current_model.cuda()
                print("FC layer loaded")

            else:
                current_model = _finetune(
                    train_data,
                    test_loader,
                    training_seed=training_seed,
                    num_epochs=num_epochs,
                    verbose=verbose,
                    output_dir=output_dir,
                    args=args,
                )
                current_model.eval()
            print("FC layer trained")

            full_train_data = data_generator.build_full_train_ssl_data(transform=None)
            train_pred_full = get_fc_scores(
                current_model, full_train_data, test_loader, data_augmentation=True, score_type=args.score_type
            )
            np.save(output_dir / f"outputs_{args.score_type}.npy", train_pred_full)
            print("Predictions saved")

            get_outputs = True
            if get_outputs:
                metrics = dict()

                print("Predicting logits and evaluating full training data")
                full_train_data = data_generator.build_full_train_data()
                train_data_full_pipe = full_train_data.as_unlabeled().build_datapipe()
                # NB: Always predict on augmented samples, even if not training with data augmentation
                train_pred_full = _predict(current_model, train_data_full_pipe, data_augmentation=True)
                # torch.save(train_pred_full, output_dir / "predictions_train.pt")

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
                print("Metrics saved")

    print("Training finished")


def _run_attack(
    args: argparse.Namespace,
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
    # ! ==============================
    num_shadow = data_generator.num_shadow

    score_file = f"outputs_{args.score_type}.npy"
    # score_file = "features_train.npy" if args.finetune == 0 else "predictions_train.npy"
    print(f"Loading shadow scores from {score_file}")
    # ! ablation study on number of random augmentations, $num_aug
    num_aug = 6
    shadow_scores_full = []
    for shadow_model_idx in range(num_shadow):
        shadow_model_dir = directory_manager.get_training_output_dir(shadow_model_idx, finetune=args.finetune)
        aug_scores = np.load(shadow_model_dir / score_file)
        avg_scores = np.mean(aug_scores[:num_aug, :, :], axis=0)
        shadow_scores_full.append(avg_scores)
    shadow_scores_full = np.array(shadow_scores_full)  # shape = (num_shadow, num_samples, num_augmentations)
    shadow_scores_full = shadow_scores_full.reshape(num_shadow, shadow_scores_full.shape[1], -1)
    shadow_scores_full = shadow_scores_full.transpose(
        1, 0, 2
    )  # (50000, 16, 18), shape = (num_samples, num_shadow, num_augmentations)
    # Only care about canaries
    scores = shadow_scores_full[canary_indices]  # shape = (num_canaries, num_shadow, num_augmentations)
    _score_name = args.score_type
    shadow_scores = {_score_name: torch.tensor(scores)}

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
                    global_variance=False,
                )
                for score_name, scores in shadow_scores.items()
            }
        else:
            shadow_attack_data = {
                score_name: attack_util.lira_attack_loo(
                    shadow_scores=scores[:, :, 0].unsqueeze(-1),
                    shadow_membership_mask=shadow_membership_mask[canary_indices],
                    global_variance=False,
                )
                for score_name, scores in shadow_scores.items()
            }

        for score_name, (scores, membership) in shadow_attack_data.items():
            print(f"## {score_name}")
            fpr, tpr = _eval_attack(
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
) -> typing.Tuple[np.ndarray, np.ndarray]:
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

    return fpr, tpr


def _get_score(
    score_type: str, model: torch.nn.Module, pos_1: torch.Tensor, pos_2: torch.Tensor, labels: torch.Tensor
) -> np.ndarray:
    score = None
    if score_type == "confidence":
        logits = model(pos_1)
        score_type = "scaled_logits"  # scaled_logits, hinge
        # ! for scaled logits
        if score_type == "scaled_logits":
            scaled_logits = logits - torch.unsqueeze(torch.max(logits, dim=1)[0], dim=1)
            scaled_logits = scaled_logits.detach().cpu().numpy()
            # get softmax output, note that the data type is float64!
            scaled_logits = np.array(np.exp(scaled_logits), dtype=np.float64)
            softmax_output = scaled_logits / np.sum(scaled_logits, axis=1, keepdims=True)
            y_true = softmax_output[np.arange(labels.size(0)), labels.detach().cpu().numpy()]
            y_wrong = 1 - y_true
            # in has higher score
            score = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
        elif score_type == "hinge":
            # use hinge loss
            target_predictions = logits[torch.arange(len(labels)), ..., labels]
            logits[torch.arange(len(labels)), ..., labels] = float("-inf")
            score = target_predictions - torch.max(logits, dim=-1).values
            score = score.detach().cpu().numpy().astype(np.float64)

    elif score_type == "similarity":
        out_1, out_2 = model(pos_1), model(pos_2)
        score = 1 * F.cosine_similarity(out_1, out_2, dim=1)  # in has higher similiarity
        # transfer to numpy as float64, and do the clip
        score = score.detach().cpu().numpy().astype(np.float64)
        score = np.clip(score, a_min=-1, a_max=1)
        score = -np.log(1 - score + 1e-45) + np.log(1 + score + 1e-45)  # in has higher score
    else:
        raise ValueError(f"wrong score type {score_type}!")
    assert score is not None
    return score


def get_fc_scores(
    model: torch.nn.Module,
    train_data: data.SSLDataset,
    test_loader: torch.utils.data.DataLoader,
    data_augmentation: bool,
    score_type: str,
) -> np.ndarray:
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )
    model.eval()
    test_acc, _ = test_accuracy(model, test_loader)
    print(f"Test accuracy: {test_acc:.2f}%")
    random_transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomResizedCrop(32, antialias=True),
            torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.v2.RandomGrayscale(p=0.2),
            torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
            torchvision.transforms.v2.Normalize(
                mean=data.CIFAR10_MEAN,
                std=data.CIFAR10_STD,
            ),
        ]
    )
    aug_features = []
    fixed_aug = 6
    for aug in tqdm.trange(fixed_aug, desc="Augmentations", unit="augmentation"):
        pred_features = np.array([])
        with torch.no_grad():
            setup_seed(aug + 721)
            for batch_xs, _, labels in train_loader:
                batch_xs = batch_xs.to(base.DEVICE)
                if not data_augmentation:
                    batch_xs_1, batch_xs_2 = random_transform(batch_xs), random_transform(batch_xs)
                    score = _get_score(score_type, model, batch_xs_1, batch_xs_2, labels)

                    if pred_features.shape[0] == 0:
                        pred_features = score
                    else:
                        pred_features = np.concatenate((pred_features, score), axis=0)
                else:
                    flip_augmentations = (False, True)
                    shift_augmentations = (0, -4, 4)
                    batch_xs_pad = torchvision.transforms.v2.functional.pad(
                        batch_xs,
                        padding=[4],
                    )
                    pred_features_current = []
                    for flip in flip_augmentations:
                        for shift_y in shift_augmentations:
                            for shift_x in shift_augmentations:
                                offset_y = shift_y + 4
                                offset_x = shift_x + 4
                                batch_xs_aug = batch_xs_pad[:, :, offset_y : offset_y + 32, offset_x : offset_x + 32]
                                if flip:
                                    batch_xs_aug = torchvision.transforms.v2.functional.hflip(batch_xs_aug)
                                batch_xs_aug_1 = random_transform(batch_xs_aug)
                                batch_xs_aug_2 = random_transform(batch_xs_aug)
                                score = _get_score(score_type, model, batch_xs_aug_1, batch_xs_aug_2, labels)
                                score = score.reshape(score.shape[0], 1)
                                pred_features_current.append(score)
                    if pred_features.shape[0] == 0:
                        pred_features = np.concatenate(pred_features_current, axis=1)
                    else:
                        pred_features = np.concatenate(
                            (pred_features, np.concatenate(pred_features_current, axis=1)), axis=0
                        )

        aug_features.append(pred_features)
    aug_features = np.array(aug_features)
    return aug_features


def get_backbone_scores(
    model: torch.nn.Module,
    train_data: data.SSLDataset,
    test_loader: torch.utils.data.DataLoader,
    memory_loader: torch.utils.data.DataLoader,
    data_augmentation: bool,
    score_type: str,
) -> np.ndarray:
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )

    model.eval()
    backbone = model.encoder_q if method == "moco" else model
    knn_test(backbone, memory_loader, test_loader)
    aug_features = []

    # do 6 augmentations by default
    random_transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomResizedCrop(32, antialias=True),
            torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.v2.RandomGrayscale(p=0.2),
            torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
            torchvision.transforms.v2.Normalize(
                mean=data.CIFAR10_MEAN,
                std=data.CIFAR10_STD,
            ),
        ]
    )
    fixed_aug = 6
    for aug in tqdm.trange(fixed_aug, desc="Augmentations", unit="augmentation"):
        pred_features = np.array([])
        with torch.no_grad():
            setup_seed(aug + 721)
            for batch_xs, _, _ in train_loader:
                batch_xs = batch_xs.to(base.DEVICE)
                if not data_augmentation:
                    batch_xs_1, batch_xs_2 = random_transform(batch_xs), random_transform(batch_xs)
                    if score_type == "backbone":
                        out_1, out_2 = model.get_backbone_embedding(batch_xs_1, batch_xs_2)
                    elif score_type == "head":
                        out_1, out_2 = model.get_head_embedding(batch_xs_1, batch_xs_2)
                    else:
                        raise ValueError(f"Wrong score type {score_type}!")
                    # out_1, out_2 = model.get_head_embedding(batch_xs_1, batch_xs_2)
                    score = 1 * F.cosine_similarity(out_1, out_2, dim=1)
                    score = score.detach().cpu().numpy().astype(np.float64)
                    score = np.clip(score, a_min=-1, a_max=1)
                    # IN has higher score, OUT has lower score
                    score = -np.log(1 - score + 1e-45) + np.log(1 + score + 1e-45)
                    score = score.reshape(score.shape[0], 1)
                    if pred_features.shape[0] == 0:
                        pred_features = score
                    else:
                        pred_features = np.concatenate((pred_features, score), axis=0)
                    # pred_features.append(score)
                else:
                    flip_augmentations = (False, True)
                    shift_augmentations = (0, -4, 4)
                    batch_xs_pad = torchvision.transforms.v2.functional.pad(
                        batch_xs,
                        padding=[4],
                    )
                    pred_features_current = []
                    for flip in flip_augmentations:
                        for shift_y in shift_augmentations:
                            for shift_x in shift_augmentations:
                                offset_y = shift_y + 4
                                offset_x = shift_x + 4
                                batch_xs_aug = batch_xs_pad[:, :, offset_y : offset_y + 32, offset_x : offset_x + 32]
                                if flip:
                                    batch_xs_aug = torchvision.transforms.v2.functional.hflip(batch_xs_aug)
                                batch_xs_aug_1 = random_transform(batch_xs_aug)
                                batch_xs_aug_2 = random_transform(batch_xs_aug)
                                if score_type == "backbone":
                                    out_1, out_2 = model.get_backbone_embedding(batch_xs_aug_1, batch_xs_aug_2)
                                elif score_type == "head":
                                    out_1, out_2 = model.get_head_embedding(batch_xs_aug_1, batch_xs_aug_2)
                                else:
                                    raise ValueError(f"Wrong score type {score_type}!")
                                # IN has higher similarity, OUT has lower similarity
                                score = 1 * F.cosine_similarity(out_1, out_2, dim=1)
                                # transfer to numpy as float64, and do the clip
                                score = score.detach().cpu().numpy().astype(np.float64)
                                score = np.clip(score, a_min=-1, a_max=1)
                                # IN has higher score, OUT has lower score
                                score = -np.log(1 - score + 1e-45) + np.log(1 + score + 1e-45)
                                score = score.reshape(score.shape[0], 1)
                                pred_features_current.append(score)

                    if pred_features.shape[0] == 0:
                        pred_features = np.concatenate(pred_features_current, axis=1)
                    else:
                        pred_features = np.concatenate(
                            (pred_features, np.concatenate(pred_features_current, axis=1)), axis=0
                        )

        aug_features.append(pred_features)
    aug_features = np.array(aug_features)  # shape: (random_augs, num_samples, fixed_augs)

    return aug_features


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

    def get_training_output_dir(self, shadow_model_idx: typing.Optional[int], finetune: int = 0) -> pathlib.Path:
        actual_suffix = "" if self._run_suffix is None else f"_{self._run_suffix}"
        if finetune == 0:
            return self._experiment_dir / ("shadow" + actual_suffix) / str(shadow_model_idx) / "backbone"
        else:
            return self._experiment_dir / ("shadow" + actual_suffix) / str(shadow_model_idx) / "fc"

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
    parser.add_argument(
        "--score_type", type=str, default=None, choices=["confidence", "similarity", "head", "backbone"]
    )

    parser.add_argument("--num-poison", type=int, default=0, help="Number of poison samples to include")
    parser.add_argument(
        "--poison-type",
        type=data.PoisonType,
        default=data.PoisonType.CANARY_DUPLICATES,
        choices=list(data.PoisonType),
        help="Type of poisoning to use",
    )
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--finetune", type=int, default=0, help="1 for finetune, 0 for not")
    parser.add_argument("--method", type=str, default="moco", choices=["simclr", "moco"], help="choose ssl methods")
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
