import typing

import torch


# Adapted from https://github.com/michaelaerni/iclr23-InductiveBiasesHarmlessInterpolation/blob/main/src/models/wide_resnet.py
# and https://github.com/facebookresearch/tan/blob/main/src/models/wideresnet.py


class WideBlock(torch.nn.Module):
    def __init__(
        self,
        features_in: int,
        features: int,
        stride: int,
        expand_features: bool,
        norm_builder: typing.Callable[
            [int, typing.Optional[torch.device], typing.Optional[torch.dtype]], torch.nn.Module
        ],
        swap_order: bool = False,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ):
        super(WideBlock, self).__init__()

        # If swap_order is True, swaps norm and relu as in DeepMind/TAN paper
        self._swap_order = swap_order

        self.norm_in = norm_builder(features_in, device, dtype)
        self.conv_in = torch.nn.Conv2d(
            features_in,
            features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.norm_out = norm_builder(features, device, dtype)
        self.conv_out = torch.nn.Conv2d(
            features,
            features,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            device=device,
            dtype=dtype,
        )

        if stride != 1 or expand_features:
            skip_conv = torch.nn.Conv2d(
                features_in,
                features,
                kernel_size=(1, 1),
                stride=stride,
                device=device,
                dtype=dtype,
            )
            if not self._swap_order:
                self.identity = skip_conv
            else:
                # NB: TAN paper use normalization before skip connections, other network does not
                self.identity = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    norm_builder(features_in, device, dtype),
                    skip_conv,
                )
        else:
            self.identity = torch.nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._swap_order:
            x = self.conv_in(torch.relu(self.norm_in(inputs)))
            x = self.conv_out(torch.relu(self.norm_out(x)))
        else:
            x = self.conv_in(self.norm_in(torch.relu(inputs)))
            x = self.conv_out(self.norm_out(torch.relu(x)))

        x += self.identity(inputs)

        return x


class WideResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        widen_factor: int,
        num_classes: int,
        use_group_norm: bool = True,
        custom_init: bool = False,
        swap_order: bool = False,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ):
        super(WideResNet, self).__init__()

        if use_group_norm:

            def norm_builder(
                num_features: int,
                device: typing.Optional[torch.device] = None,
                dtype: typing.Optional[torch.dtype] = None,
            ):
                return torch.nn.GroupNorm(min(16, num_features), num_features, device=device, dtype=dtype)

        else:

            def norm_builder(
                num_features: int,
                device: typing.Optional[torch.device] = None,
                dtype: typing.Optional[torch.dtype] = None,
            ):
                return torch.nn.BatchNorm2d(num_features, device=device, dtype=dtype)

        # If swap_order is True, swaps norm and relu as in DeepMind/TAN paper
        self._swap_order = swap_order

        assert (depth - 4) % 6 == 0 and depth > 4
        assert widen_factor > 0
        num_blocks = (depth - 4) // 6
        features_per_block = (
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        )

        features_in = features_per_block[0]
        self.conv_in = torch.nn.Conv2d(
            in_channels,
            features_in,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=device,
            dtype=dtype,
        )

        self.group_1, features_in = self._wide_layer(
            features_in,
            features_per_block[1],
            stride=1,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            swap_order=self._swap_order,
            device=device,
            dtype=dtype,
        )
        self.group_2, features_in = self._wide_layer(
            features_in,
            features_per_block[2],
            stride=2,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            swap_order=self._swap_order,
            device=device,
            dtype=dtype,
        )
        self.group_3, features_in = self._wide_layer(
            features_in,
            features_per_block[3],
            stride=2,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            swap_order=self._swap_order,
            device=device,
            dtype=dtype,
        )

        self.norm_out = norm_builder(features_in, device=device, dtype=dtype)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()

        self._dense_features_in = features_in
        self._custom_init = custom_init
        self.dense = torch.nn.Linear(self._dense_features_in, num_classes, bias=True, device=device, dtype=dtype)

        if custom_init:
            # Taken from https://github.com/facebookresearch/tan/blob/main/src/models/wideresnet.py
            # See license in dpsgd_utils.py
            assert use_group_norm
            for module in self.modules():
                if isinstance(module, torch.nn.Conv2d):
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                    std = 1 / (max(fan_in, 1)) ** 0.5
                    torch.nn.init.trunc_normal_(module.weight, std=std)
                    module.bias.data.zero_()
                elif isinstance(module, torch.nn.GroupNorm):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, torch.nn.Linear):
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                    std = 1 / (max(fan_in, 1)) ** 0.5
                    torch.nn.init.trunc_normal_(module.weight, std=std)
                    module.bias.data.zero_()

    def replace_dense(self, num_classes: int):
        device = self.dense.weight.device
        dtype = self.dense.weight.dtype
        self.dense = torch.nn.Linear(self._dense_features_in, num_classes, bias=True, device=device, dtype=dtype)

        # Redo initialization
        if self._custom_init:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
            std = 1 / (max(fan_in, 1)) ** 0.5
            torch.nn.init.trunc_normal_(self.dense.weight, std=std)
            self.dense.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        x = self.group_1(x)
        x = self.group_2(x)
        x = self.group_3(x)

        if not self._swap_order:
            x = torch.relu(self.norm_out(x))
        else:
            x = self.norm_out(torch.relu(x))

        # Pool and flatten
        x = self.flatten(self.pool(x))

        x = self.dense(x)

        return x

    @staticmethod
    def _wide_layer(
        features_in: int,
        features: int,
        num_blocks: int,
        stride: int,
        norm_builder: typing.Callable[
            [int, typing.Optional[torch.device], typing.Optional[torch.dtype]], torch.nn.Module
        ],
        swap_order: bool,
        device: typing.Optional[torch.device] = None,
        dtype: typing.Optional[torch.dtype] = None,
    ) -> typing.Tuple[torch.nn.Module, int]:
        blocks = []
        for block_idx in range(num_blocks):
            block = WideBlock(
                features_in=features_in,
                features=features,
                stride=stride if block_idx == 0 else 1,
                expand_features=(features_in != features),
                norm_builder=norm_builder,
                swap_order=swap_order,
                device=device,
                dtype=dtype,
            )
            blocks.append(block)
            features_in = features

        return torch.nn.Sequential(*blocks), features_in
