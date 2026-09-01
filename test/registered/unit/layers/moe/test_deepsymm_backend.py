"""CPU-only tests for the DeepSymm compatibility seam."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.quantization.unquant import (
    _xpu_combine_input,
    _xpu_dispatch_tensors,
)
from sglang.srt.layers.quantization.mxfp4 import (
    _xpu_mxfp4_combine_input,
    _xpu_mxfp4_dispatch_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_pcie_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes * num_ranks


class _FakeBuffer:
    num_eus = 20

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeCudaConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_nvl_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes * num_ranks

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes + num_ranks


class _FakeCudaBuffer:
    num_sms = 132

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestDeepSymmBackend(CustomTestCase):
    def test_backend_is_part_of_deepep_dispatcher_family(self):
        backend = moe_utils.MoeA2ABackend("deepsymm")
        self.assertTrue(backend.is_deepsymm())
        self.assertFalse(backend.is_deepep())
        self.assertTrue(backend.uses_deepep_dispatcher())

        deepep_backend = moe_utils.MoeA2ABackend("deepep")
        self.assertTrue(deepep_backend.is_deepep())
        self.assertFalse(deepep_backend.is_deepsymm())
        self.assertTrue(deepep_backend.uses_deepep_dispatcher())

    def test_xpu_config_translates_deepep_names(self):
        values = {
            "num_sms": 20,
            "num_max_nvl_chunked_send_tokens": 6,
            "num_max_nvl_chunked_recv_tokens": 256,
            "num_max_rdma_chunked_send_tokens": 6,
            "num_max_rdma_chunked_recv_tokens": 128,
        }
        with (
            patch.object(deepep, "_is_xpu", True),
            patch.object(deepep, "Config", _FakeConfig, create=True),
        ):
            config = deepep._create_config(values)

        self.assertEqual(
            config.kwargs,
            {
                "num_eus": 20,
                "num_max_pcie_chunked_send_tokens": 6,
                "num_max_pcie_chunked_recv_tokens": 256,
                "num_max_rdma_chunked_send_tokens": 6,
                "num_max_rdma_chunked_recv_tokens": 128,
            },
        )

    def test_xpu_buffer_uses_pcie_hint_and_drops_mnnvl(self):
        config = _FakeConfig()
        with (
            patch.object(deepep, "_is_xpu", True),
            patch.object(deepep, "Buffer", _FakeBuffer, create=True),
        ):
            self.assertEqual(deepep._get_comm_unit_count(), 20)
            self.assertEqual(deepep._get_local_buffer_size_hint(config, 8, 4), 32)
            self.assertEqual(deepep._get_rdma_buffer_size_hint(config, 8, 4), 0)
            buffer = deepep._create_buffer(
                object(),
                1024,
                0,
                low_latency_mode=False,
                allow_mnnvl=False,
            )

        self.assertEqual(buffer.args[1:], (1024, 0))
        self.assertNotIn("allow_mnnvl", buffer.kwargs)

    def test_cuda_deepep_config_and_buffer_api_remain_unchanged(self):
        values = {
            "num_sms": 24,
            "num_max_nvl_chunked_send_tokens": 6,
            "num_max_nvl_chunked_recv_tokens": 256,
        }
        config = _FakeCudaConfig()
        with (
            patch.object(deepep, "_is_xpu", False),
            patch.object(deepep, "Config", _FakeCudaConfig, create=True),
            patch.object(deepep, "Buffer", _FakeCudaBuffer, create=True),
        ):
            created_config = deepep._create_config(values)
            self.assertEqual(deepep._get_comm_unit_count(), 132)
            self.assertEqual(deepep._get_local_buffer_size_hint(config, 8, 4), 32)
            self.assertEqual(deepep._get_rdma_buffer_size_hint(config, 8, 4), 12)
            buffer = deepep._create_buffer(
                object(),
                1024,
                2048,
                low_latency_mode=False,
                allow_mnnvl=True,
            )

        self.assertEqual(created_config.kwargs, values)
        self.assertEqual(buffer.args[1:], (1024, 2048))
        self.assertTrue(buffer.kwargs["allow_mnnvl"])

    def test_xpu_native_moe_preserves_deepsymm_combine_metadata(self):
        hidden_states = torch.randn(3, 8)
        topk_ids = torch.tensor([[0, -1], [1, 2], [-1, 3]])
        topk_weights = torch.rand(3, 2)
        dispatch_output = deepep.DeepEPNormalDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[1, 1, 1, 1],
        )

        dispatch_hidden_states, dispatch_weights, dispatch_ids = _xpu_dispatch_tensors(
            dispatch_output
        )
        self.assertIs(dispatch_hidden_states, hidden_states)
        torch.testing.assert_close(
            dispatch_weights,
            topk_weights * (topk_ids >= 0),
        )
        torch.testing.assert_close(dispatch_ids, topk_ids.clamp_min(0))
        output = torch.randn_like(hidden_states)
        combine_input = _xpu_combine_input(output, dispatch_output)
        self.assertIs(combine_input.hidden_states, output)
        self.assertIs(combine_input.topk_ids, topk_ids)
        self.assertIs(combine_input.topk_weights, topk_weights)

    def test_xpu_mxfp4_preserves_deepsymm_combine_metadata(self):
        hidden_states = torch.randn(2, 8)
        topk_ids = torch.tensor([[0, -1], [2, 3]])
        topk_weights = torch.rand(2, 2)
        dispatch_output = deepep.DeepEPNormalDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[1, 0, 1, 1],
        )

        x, dispatch_weights, dispatch_ids, is_deepep_normal = (
            _xpu_mxfp4_dispatch_tensors(dispatch_output)
        )
        self.assertIs(x, hidden_states)
        self.assertTrue(is_deepep_normal)
        torch.testing.assert_close(dispatch_ids, torch.tensor([[0, 0], [2, 3]]))
        torch.testing.assert_close(
            dispatch_weights,
            topk_weights * torch.tensor([[1, 0], [1, 1]]),
        )

        output = torch.randn_like(hidden_states)
        combine_input = _xpu_mxfp4_combine_input(
            output, dispatch_output, is_deepep_normal
        )
        self.assertIs(combine_input.hidden_states, output)
        self.assertIs(combine_input.topk_ids, topk_ids)
        self.assertIs(combine_input.topk_weights, topk_weights)


if __name__ == "__main__":
    unittest.main()
