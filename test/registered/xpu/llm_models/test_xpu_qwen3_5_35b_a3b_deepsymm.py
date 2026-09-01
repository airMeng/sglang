"""DeepSymm phase-1 model scope on Intel XPU.

The gate covers single-node 4-XPU BF16 serving with expert parallelism across
all four ranks. Qwen3.5-35B-A3B-L20 exercises 256 routed experts, top-8 routing,
and hidden size 2048. Phase 1 intentionally excludes low-latency and internode
transport, quantized dispatch, speculative decoding, and XPU graph capture.
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=2400, suite="nightly-xpu-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_5_35BA3BDeepSymmXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = os.getenv("SGLANG_DEEPSYMM_TEST_MODEL", "/data/model/Qwen3.5-35B-A3B-L20")
    tp_size = 4
    accuracy = 0.90
    timeout_for_server_launch = 3600
    num_examples = 50
    num_threads = 4
    max_tokens = 8192
    env = {"SGLANG_USE_SGL_XPU": "1"}

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--json-model-override-args",
        '{"language_model_only":true}',
        "--skip-server-warmup",
        "--ep-size",
        "4",
        "--moe-a2a-backend",
        "deepsymm",
        "--deepep-mode",
        "normal",
        "--page-size",
        "128",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.85",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]


if __name__ == "__main__":
    unittest.main()
