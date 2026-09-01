"""GPT-OSS-20B MXFP4 DeepSymm parity smoke test on four Intel XPUs.

The scope is single-node TP=4/EP=4, DeepSymm normal mode, native
sgl-kernel-xpu W4A16 experts, eager prefill/decode, and deterministic text
parity with the standard TP=4 backend. Internode, low-latency, and graph
capture are intentionally out of scope.
"""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=600, suite="nightly-xpu-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestGptOss20BMxfp4DeepSymmXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.getenv(
            "SGLANG_DEEPSYMM_MXFP4_TEST_MODEL", "/data/model/gpt-oss-20b-mxfp4"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=1800,
            device="xpu",
            env={"SGLANG_USE_SGL_XPU": "1"},
            other_args=[
                "--attention-backend",
                "intel_xpu",
                "--dtype",
                "bfloat16",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
                "--skip-server-warmup",
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--moe-a2a-backend",
                "deepsymm",
                "--deepep-mode",
                "normal",
                "--page-size",
                "64",
                "--max-total-tokens",
                "4096",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_matches_standard_tp4_greedy_tokens(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital city of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=240,
        )
        response.raise_for_status()
        result = response.json()
        self.assertEqual(
            result["output_ids"],
            [12650, 13, 279, 976, 9029, 5030, 328, 10128],
        )
        self.assertTrue(result["text"].startswith(" Paris."))


if __name__ == "__main__":
    unittest.main()
