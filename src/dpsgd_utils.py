# This code is mostly copied from https://github.com/facebookresearch/tan/ (w/ slight modifications)

# The original code is licensed under the BSD 3-Clause License:

# BSD 3-Clause License
#
# Copyright (c) 2022, Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import collections
import copy
import os
import warnings
from functools import partial
from typing import IO, Any, BinaryIO, Dict, Optional, Union
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.layers.dp_rnn import DPRNNBase, DPRNNCellBase, RNNLinear
from opacus.optimizers import DPOptimizer, get_optimizer_class
from opacus.schedulers import _NoiseScheduler
from opacus.utils.module_utils import (
    requires_grad,
    trainable_modules,
)
from opacus.utils.tensor_utils import unfold2d, sum_over_all_but_batch_and_last_n
from opacus.validators.module_validator import ModuleValidator
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


class GradSampleModuleAugmented(nn.Module):

    r"""
    Extends nn.Module so that its parameter tensors have an extra field called .grad_sample.
    """

    GRAD_SAMPLERS = {}

    def __init__(
        self, m: nn.Module, GRAD_SAMPLERS_, *, batch_first=True, loss_reduction="mean", strict: bool = True, K: int = 0
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to ``True``, the input module will be validated to check that
                ``GradSampleModule`` has grad sampler functions for all submodules of
                the input module (i.e. if it knows how to calculate per sample gradients)
                for all model parameters. If set to ``False``, per sample gradients will
                be computed on "best effort" basis - they will be available where
                possible and set to None otherwise. This is not recommended, because
                some unsupported modules (e.g. BatchNorm) affect other parameters and
                invalidate the concept of per sample gradients for the entire model.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) doesn't have a registered grad sampler function.
        """

        GRAD_SAMPLERS = GRAD_SAMPLERS_
        super().__init__()
        self.GRAD_SAMPLERS = GRAD_SAMPLERS

        errors = self.validate(module=m, strict=strict)
        if errors and not strict:
            print(f"GradSampleModule found the following errors: {errors}." "Using non-strict mode, continuing")

        self._module = m
        self.hooks_enabled = False
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.add_hooks(loss_reduction=loss_reduction, batch_first=batch_first, K=K)

        for _, p in trainable_parameters(self):
            p.grad_sample = None
            p._forward_counter = 0

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            submodules = dict(self._module.named_modules())
            if item and item in submodules:
                return submodules[item]
            raise e

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad`` and ``p.grad_sample`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` is
            never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """
        if set_to_none is False:
            print(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample to None due to "
                "non-trivial gradient accumulation behaviour"
            )
        self.set_grad_sample_to_none()
        super().zero_grad(set_to_none)

    def set_grad_sample_to_none(self):
        """
        Sets ``.grad_sample`` to None
        """
        for _, p in trainable_parameters(self):
            p.grad_sample = None

    def del_grad_sample(self):
        """
        Deleted ``.grad_sample`` attribute from all model parameters
        """
        for _, p in trainable_parameters(self):
            del p.grad_sample

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self._close()
        return self._module

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
        K: int = 0,
    ) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for _module_name, module in trainable_modules(self._module):
            if type(module) in self.GRAD_SAMPLERS:
                self.autograd_grad_sample_hooks.append(module.register_forward_hook(self.capture_activations_hook))

                self.autograd_grad_sample_hooks.append(
                    module.register_backward_hook(
                        partial(
                            self.capture_backprops_hook, loss_reduction=loss_reduction, batch_first=batch_first, K=K
                        )
                    )
                )
        self.enable_hooks()

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        for p in self.parameters():
            if hasattr(p, "ddp_hooks"):
                while p.ddp_hooks:
                    handle = p.ddp_hooks.pop()
                    handle.remove()
                delattr(p, "ddp_hooks")

        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.
        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        r"""
        The opposite of ``disable_hooks()``. Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        self.hooks_enabled = True

    def __repr__(self):
        return f"GradSampleModule({self._module.__repr__()})"

    def _close(self):
        self.del_grad_sample()
        self.remove_hooks()
        self._clean_up_attributes()

    def _clean_up_attributes(self):
        for attr in OPACUS_PARAM_MONKEYPATCH_ATTRS:
            for p in self.parameters():
                if hasattr(p, attr):
                    delattr(p, attr)

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if not requires_grad(module) or not module.training or not torch.is_grad_enabled():
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append(forward_input[0].detach())  # pyre-ignore

        for _, p in trainable_parameters(module):
            p._forward_counter += 1

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
        K: int,
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradients are
        stored in ``grad_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample``.

        From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
        if accumulated over multiple batches

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module=module, backprops=backprops, loss_reduction=loss_reduction, batch_first=batch_first, K=K
        )
        grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        grad_samples = grad_sampler_fn(module, activations, backprops)
        for param, gs in grad_samples.items():
            create_or_accumulate_grad_sample(param=param, grad_sample=gs, max_batch_len=module.max_batch_len)

        # Detect end of current batch processing and switch accumulation
        # mode from sum to stacking. Used for RNNs and tied parameters
        # (See #417 for details)
        for _, p in trainable_parameters(module):
            p._forward_counter -= 1
            if p._forward_counter == 0:
                promote_current_grad_sample(p)

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def rearrange_grad_samples(
        self,
        *,
        module: nn.Module,
        backprops: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rearrange activations and grad_samples based on loss reduction and batch dim

        Args:
            module: the module for which per-sample gradients are computed
            backprops: the captured backprops
            loss_reduction: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch
            batch_first: True is batch dimension is first
        """
        if not hasattr(module, "activations"):
            raise ValueError(f"No activations detected for {type(module)}," " run forward after add_hooks(model)")

        batch_dim = 0 if batch_first or type(module) is RNNLinear else 1

        activations = module.activations.pop()

        if not hasattr(module, "max_batch_len"):
            # For packed sequences, max_batch_len is set in the forward of the model (e.g. the LSTM)
            # Otherwise we infer it here
            module.max_batch_len = _get_batch_size(
                module=module,
                grad_sample=activations,
                batch_dim=batch_dim,
                K=K,
            )

        n = module.max_batch_len
        if loss_reduction == "mean":
            backprops = backprops * n
        elif loss_reduction == "sum":
            backprops = backprops
        else:
            raise ValueError(f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported")

        # No matter where the batch dimension was, .grad_samples will *always* put it in the first dim
        if batch_dim != 0:
            activations = activations.permute([batch_dim] + [x for x in range(activations.dim()) if x != batch_dim])
            backprops = backprops.permute([batch_dim] + [x for x in range(backprops.dim()) if x != batch_dim])

        return activations, backprops

    @classmethod
    def is_supported(cls, module: nn.Module) -> bool:
        """
        Checks if this individual model is supported (i.e. has a registered
        grad sampler function)

        Notes:
            Note that this method does not check submodules

        Args:
            module: nn.Module to be checked

        Returns:
            ``True`` if grad sampler is found, ``False`` otherwise
        """
        return type(module) in cls.GRAD_SAMPLERS or isinstance(module, (DPRNNBase, DPRNNCellBase))

    @classmethod
    def validate(cls, module: nn.Module, *, strict: bool = False) -> List[NotImplementedError]:
        """
        Check if per sample gradients can be fully computed for a given model

        Args:
            module: nn.Module to be checked
            raise_if_error: Behaviour in case of a negative check result. Will
            return the list of exceptions if set to ``False``, and throw otherwise

        Returns:
            Empty list of validation is successful.
            List of validation errors  if ``raise_if_error=False`` and
            unsupported modules are found

        Raises:
            NotImplementedError
                If ``raise_if_error=True`` and unsupported modules are found
        """
        errors = []
        errors.extend(
            [
                NotImplementedError(
                    f"Model contains a trainable layer "
                    f"that Opacus doesn't currently support({m_name}:{m}). "
                    f"Please implement and register grad sampler for this layer. "
                    f"(See opacus.grad_sample.utils.register_grad_sampler)"
                )
                for m_name, m in trainable_modules(module)
                if not GradSampleModuleAugmented.is_supported(m)
            ]
        )
        # raise or return errors as needed
        if strict and len(errors) > 0:
            raise NotImplementedError(errors)
        else:
            return errors


def forbid_accumulation_hook(module: GradSampleModuleAugmented, _grad_input: torch.Tensor, _grad_output: torch.Tensor):
    """
    Model hook that detects repetitive forward/backward passes between optimizer steps.

    This is a backward hook that will be wrapped around the whole model using
    `register_backward_hook`. We wish to detect a case where:
        -  `optimizer.zero_grad()` is not called before the backward pass; and
        -  `p.grad_sample` was updated in a *previous* iteration.

    To do so, we attach a backward hook to the model that runs *after* the computation
    of `grad_sample` for the current step. We compute the number of accumulated iterations
    like on `optimizers/optimizer.py` and check whether it's strictly larger than one.

    Args:
        module: input module
        _grad_input: module input gradient (not used here)
        _grad_output: module output gradient (not used here)

    Raises:
        ValueError
            If the hook detected multiple forward/backward passes between optimizer steps

    """
    if not module.training:
        return

    for _, p in trainable_parameters(module):
        if p.grad_sample is not None:
            if isinstance(p.grad_sample, torch.Tensor):
                accumulated_iterations = 1
            elif isinstance(p.grad_sample, list):
                accumulated_iterations = len(p.grad_sample)

            if accumulated_iterations > 1:
                raise ValueError(
                    "Poisson sampling is not compatible with grad accumulation. "
                    "You need to call optimizer.step() after every forward/backward pass "
                    "or consider using BatchMemoryManager"
                )


class PrivacyEngineAugmented:
    """
    Main entry point to the Opacus API - use ``PrivacyEngineAugmented``  to enable differential
    privacy for your model training.

    ``PrivacyEngineAugmented`` object encapsulates current privacy state (privacy budget +
    method it's been calculated) and exposes ``make_private`` method to wrap your
    PyTorch training objects with their private counterparts.

    Example:
        >>> dataloader = demo_dataloader
        >>> model = MyCustomModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> privacy_engine = PrivacyEngineAugmented()
        >>>
        >>> model, optimizer, dataloader = privacy_engine.make_private(
        ...    module=model,
        ...    optimizer=optimizer,
        ...    data_loader=dataloader,
        ...    noise_multiplier=1.0,
        ...    max_grad_norm=1.0,
        ... )
        >>> # continue training as normal
    """

    def __init__(self, GRAD_SAMPLERS, *, accountant: str = "rdp", secure_mode: bool = False):
        """

        Args:
            accountant: Accounting mechanism. Currently supported:
                - rdp (:class:`~opacus.accountants.RDPAccountant`)
                - gdp (:class:`~opacus.accountants.GaussianAccountant`)
            secure_mode: Set to ``True`` if cryptographically strong DP guarantee is
                required. ``secure_mode=True`` uses secure random number generator for
                noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch) and
                prevents certain floating-point arithmetic-based attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details.
                When set to ``True`` requires ``torchcsprng`` to be installed
        """
        GradSampleModuleAugmented.GRAD_SAMPLERS = GRAD_SAMPLERS

        self.GRAD_SAMPLERS = GRAD_SAMPLERS
        self.accountant = create_accountant(mechanism=accountant)
        self.secure_mode = secure_mode
        self.secure_rng = None
        self.dataset = None  # only used to detect switching to a different dataset
        if self.secure_mode:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_mode`` turned on."
            )

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(clipping=clipping, distributed=distributed)

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
    ) -> DataLoader:
        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(
                f"PrivacyEngineAugmented detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngineAugmented if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset"
            )

        if poisson_sampling:
            return DPDataLoader.from_data_loader(data_loader, generator=self.secure_rng, distributed=distributed)
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        K=0,
    ) -> GradSampleModuleAugmented:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, GradSampleModuleAugmented):
            if module.batch_first != batch_first or module.loss_reduction != loss_reduction:
                raise ValueError(
                    f"Pre-existing GradSampleModuleAugmented doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}. "
                    f"Please pass vanilla nn.Module instead"
                )

            return module
        else:
            ret = GradSampleModuleAugmented(
                module, self.GRAD_SAMPLERS, batch_first=batch_first, loss_reduction=loss_reduction, K=K
            )
            return ret

    def is_compatible(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        """
        Check if task components are compatible with DP.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Returns:
            ``True`` if compatible, ``False`` otherwise
        """
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        """
        Validate that task components are compatible with DP.
        Same as ``is_compatible()``, but raises error instead of returning bool.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Raises:
            UnsupportedModuleError
                If one or more modules found to be incompatible
        """
        ModuleValidator.validate(module, strict=True)

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        """
        Return a privacy engine compatible module. Also validates the module after
        running registered fixes.

        Args:
            module: module to be modified

        Returns:
            Module with some submodules replaced for their deep copies or
            close equivalents.
            See :class:`~opacus.validators.module_validator.ModuleValidator` for
            more details
        """
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, strict=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        K=0,
    ) -> Tuple[GradSampleModuleAugmented, DPOptimizer, DataLoader]:
        """
        Add privacy-related responsibilites to the main PyTorch training objects:
        model, optimizer, and the data loader.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
             gradient clipping and noise addition to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """

        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        # if not all(
        #     torch.eq(i, j).all()
        #     for i, j in zip(
        #         list(module.parameters()),
        #         sum(
        #             [param_group["params"] for param_group in optimizer.param_groups],
        #             [],
        #         ),
        #     )
        # ):
        #     raise ValueError(
        #         "Module parameters are different than optimizer Parameters"
        #     )

        distributed = isinstance(module, (DPDDP, DDP))

        module = self._prepare_model(module, batch_first=batch_first, loss_reduction=loss_reduction, K=K)

        if poisson_sampling:
            module.register_backward_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(data_loader, distributed=distributed, poisson_sampling=poisson_sampling)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
        )

        optimizer.attach_step_hook(self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_generator=None,
        **kwargs,
    ):
        """
        Version of :meth:`~opacus.privacy_engine.PrivacyEngineAugmented.make_private`,
        that calculates privacy parameters based on a given privacy budget.

        For the full documentation see
        :meth:`~opacus.privacy_engine.PrivacyEngineAugmented.make_private`

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            noise_seed: Seed to be used for random noise generation
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
             gradient clipping and adding noise to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        sample_rate = 1 / len(data_loader)

        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant=self.accountant.mechanism(),
                **kwargs,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
        )

    def get_epsilon(self, delta):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        Args:
            delta: The target delta.

        Returns:
            Privacy budget (epsilon) expended so far.
        """
        return self.accountant.get_epsilon(delta)

    def save_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModuleAugmented,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        module_state_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_save_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves the state_dict of module, optimzer, and accountant at path.
        Args:
            path: Path to save the state dict objects.
            module: GradSampleModuleAugmented to save; wrapped module's state_dict is saved.
            optimizer: DPOptimizer to save; wrapped optimizer's state_dict is saved.
            module_state_dict_kwargs: dict of kwargs to pass to ``module.state_dict()``
            torch_save_kwargs: dict of kwargs to pass to ``torch.save()``

        """
        checkpoint_dict = checkpoint_dict or {}
        checkpoint_dict["module_state_dict"] = module.state_dict(**(module_state_dict_kwargs or {}))
        checkpoint_dict["privacy_accountant_state_dict"] = self.accountant.state_dict()
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        if noise_scheduler is not None:
            checkpoint_dict["noise_scheduler_state_dict"] = noise_scheduler.state_dict()

        torch.save(checkpoint_dict, path, **(torch_save_kwargs or {}))

    def load_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModuleAugmented,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        module_load_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        checkpoint = torch.load(path, **(torch_load_kwargs or {}))
        module.load_state_dict(checkpoint["module_state_dict"], **(module_load_dict_kwargs or {}))
        self.accountant.load_state_dict(checkpoint["privacy_accountant_state_dict"])

        optimizer_state_dict = checkpoint.pop("optimizer_state_dict", {})
        if optimizer is not None and len(optimizer_state_dict) > 0:
            optimizer.load_state_dict(optimizer_state_dict)
        elif (optimizer is not None) ^ (len(optimizer_state_dict) > 0):
            # warn if only one of them is available
            warnings.warn(
                f"optimizer_state_dict has {len(optimizer_state_dict)} items"
                f" but optimizer is {'' if optimizer else 'not'} provided."
            )

        noise_scheduler_state_dict = checkpoint.pop("noise_scheduler_state_dict", {})
        if noise_scheduler is not None and len(noise_scheduler_state_dict) > 0:
            noise_scheduler.load_state_dict(noise_scheduler_state_dict)

        return checkpoint


OPACUS_PARAM_MONKEYPATCH_ATTRS = ["_forward_counter", "_current_grad_sample"]


def create_or_accumulate_grad_sample(*, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int) -> None:
    """
    Creates a ``_current_grad_sample`` attribute in the given parameter, or adds to it
    if the ``_current_grad_sample`` attribute already exists.


    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        layer: nn.Module parameter belongs to
    """
    if param.requires_grad:
        if hasattr(param, "_current_grad_sample"):
            param._current_grad_sample[: grad_sample.shape[0]] += grad_sample
        else:
            param._current_grad_sample = torch.zeros(
                torch.Size([max_batch_len]) + grad_sample.shape[1:],
                device=grad_sample.device,
                dtype=grad_sample.dtype,
            )
            # if param._current_grad_sample[: grad_sample.shape[0]].shape[1:]==(197, 384):
            #     import pdb;pdb.set_trace()
            param._current_grad_sample[: grad_sample.shape[0]] = grad_sample


def promote_current_grad_sample(p: nn.Parameter) -> None:
    if p.requires_grad:
        if p.grad_sample is not None:
            if isinstance(p.grad_sample, list):
                p.grad_sample.append(p._current_grad_sample)
            else:
                p.grad_sample = [p.grad_sample, p._current_grad_sample]
        else:
            p.grad_sample = p._current_grad_sample

        del p._current_grad_sample


def _get_batch_size(*, module: nn.Module, grad_sample: torch.Tensor, batch_dim: int, K: int) -> int:
    """
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations + [grad_sample], where module.activations is
    a list.

    If module.activations is a not a list, then return grad_sample.shape[batch_dim].

    Args:
        module: input module
        grad_sample: per sample gradient tensor
        batch_dim: batch dimension

    Returns:
        Maximum sequence length in a batch
    """

    max_batch_len = 0
    for out in module.activations:
        if out.shape[batch_dim] > max_batch_len:
            max_batch_len = out.shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len if not (K) else max_batch_len // K


class AugmentationMultiplicity:
    def __init__(self, K):
        self.K = K

    def augmented_compute_conv_grad_sample(
        self,
        layer: nn.Conv2d,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for convolutional layers
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        n = activations.shape[0]
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        backprops = backprops.reshape(n, -1, activations.shape[-1])

        ret = {}
        if layer.weight.requires_grad:
            # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
            activations = activations.reshape(
                (
                    -1,
                    self.K,
                )
                + (activations.shape[1:])
            )
            backprops = backprops.reshape(
                (
                    -1,
                    self.K,
                )
                + (backprops.shape[1:])
            )
            grad_sample = torch.einsum("nkoq,nkpq->nop", backprops, activations)
            # rearrange the above tensor and extract diagonals.
            n = activations.shape[0]
            grad_sample = grad_sample.view(
                n,
                layer.groups,
                -1,
                layer.groups,
                int(layer.in_channels / layer.groups),
                np.prod(layer.kernel_size),
            )
            grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
            shape = [n] + list(layer.weight.shape)
            ret[layer.weight] = grad_sample.view(shape)

        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("nkoq->no", backprops)

        return ret

    def augmented_compute_expand_grad_sample(self, layer, activations, backprops):
        """
        Computes per sample gradients for expand layers.
        """
        return {layer.weight: backprops.reshape((-1, self.K) + (backprops.shape[1:])).sum(1)}

    def augmented_compute_linear_grad_sample(
        self, layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for ``nn.Linear`` layer
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        activations = activations.reshape(
            (
                -1,
                self.K,
            )
            + (activations.shape[1:])
        )
        backprops = backprops.reshape(
            (
                -1,
                self.K,
            )
            + (backprops.shape[1:])
        )
        if layer.weight.requires_grad:
            gs = torch.einsum("n...i,n...j->nij", backprops, activations)
            ret[layer.weight] = gs
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...k->nk", backprops)
        return ret

    def augmented_compute_group_norm_grad_sample(
        self,
        layer: nn.GroupNorm,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for GroupNorm
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            normalize_activations = F.group_norm(activations, layer.num_groups, eps=layer.eps)
            normalize_activations = normalize_activations.reshape(
                (
                    -1,
                    self.K,
                )
                + (activations.shape[1:])
            )
            backprops = backprops.reshape(
                (
                    -1,
                    self.K,
                )
                + (backprops.shape[1:])
            )
            ret[layer.weight] = torch.einsum("nki..., nki...->ni", normalize_activations, backprops)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("nki...->ni", backprops)
        return ret

    def augmented_compute_layer_norm_grad_sample(
        self,
        layer: nn.LayerNorm,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for LayerNorm
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
            normalize_activations = normalize_activations.reshape(
                (
                    -1,
                    self.K,
                )
                + (activations.shape[1:])
            )
            backprops = backprops.reshape(
                (
                    -1,
                    self.K,
                )
                + (backprops.shape[1:])
            )
            ret[layer.weight] = sum_over_all_but_batch_and_last_n(
                normalize_activations * backprops,
                layer.weight.dim(),
            )
        if layer.bias.requires_grad:
            ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
        return ret


def trainable_parameters(module):
    """
    Recursively iterates over all parameters, returning those that
    are trainable (ie they want a grad).
    """
    yield from ((p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad)


def create_ema(model: torch.nn.Module) -> torch.nn.Module:
    ema = copy.deepcopy(model)
    for param in ema.parameters():
        param.detach_()
    return ema


def update_ema(model: torch.nn.Module, ema: torch.nn.Module, t, decay=0.9999, change_ema_decay_end=0):
    t2 = t - change_ema_decay_end if t > change_ema_decay_end else t

    effective_decay = min(decay, (1 + t2) / (10 + t2))
    model_params = collections.OrderedDict(model.named_parameters())
    ema_params = collections.OrderedDict(ema.named_parameters())
    # check if both model contains the same set of keys
    assert model_params.keys() == ema_params.keys()

    for name, param in model_params.items():
        # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # ema_variable -= (1 - decay) * (ema_variable - variable)
        ema_params[name].sub_((1.0 - effective_decay) * (ema_params[name] - param))

    model_buffers = collections.OrderedDict(model.named_buffers())
    ema_buffers = collections.OrderedDict(ema.named_buffers())

    # check if both model contains the same set of keys
    assert model_buffers.keys() == ema_buffers.keys()

    for name, buffer in model_buffers.items():
        # buffers are copied
        ema_buffers[name].copy_(buffer)
