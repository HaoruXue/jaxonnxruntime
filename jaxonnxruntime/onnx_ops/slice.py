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

"""Define ONNX Slice operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
import jax.numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op('Slice')
class Slice(handler.Handler):
  """Implementation of the ONNX Slice operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Slice op."""
    cls._prepare(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 Slice op."""
    cls._prepare(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Slice op."""
    cls._prepare(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Slice op."""
    cls._prepare(node, inputs, onnx_slice)
    return onnx_slice


@functools.partial(jax.jit, static_argnames=())
def onnx_slice(*input_args):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Slice."""
  x, starts, ends = input_args[:3]
  assert len(starts) == len(ends)

  if len(input_args) < 3:
    axes = None
    assert len(starts) == len(x.shape)
    lax_start = starts
    lax_end = ends
  else:
    axes = input_args[3]
    lax_start = jnp.zeros(len(x.shape), dtype=jnp.int32)
    lax_end = jnp.array(x.shape, dtype=jnp.int32)
    lax_start = lax_start.at[axes].set(starts)
    lax_end = lax_end.at[axes].set(ends)

  if len(input_args) > 4:
    steps = input_args[4]
    lax_steps = jnp.ones(len(x.shape), dtype=jnp.int32)
    lax_steps = lax_steps.at[axes].set(steps)
  else:
    steps = None

  return jax.lax.slice(x, lax_start, lax_end, lax_steps)
