# Copyright 2024 The Jaxonnxruntime Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define ONNX Mod operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Mod")
class Mod(handler.Handler):
  """Implementation of the ONNX Mod operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)
    node.attrs_dict["fmod"] = node.attrs.get("fmod", 0)

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 Mod op."""
    cls._prepare(node, inputs, onnx_mod)
    return onnx_mod

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Mod op."""
    cls._prepare(node, inputs, onnx_mod)
    return onnx_mod


@functools.partial(jax.jit, static_argnames=("fmod",))
def onnx_mod(*input_args, fmod):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#mod."""
  assert len(input_args) == 2
  a, b = input_args
  if fmod == 0:
    return jnp.mod(a, b)
  return jnp.fmod(a, b)
