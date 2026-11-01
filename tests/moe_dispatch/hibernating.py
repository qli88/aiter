# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GPU test of each hibernating segment (`reach = "hibernating"`).

No ranking entry and no shipped tuned row runs a hibernating segment, so only its test
keeps it from breaking unnoticed. `coverage_checks.py --check` fails a hibernating
segment missing here, and an entry whose test is not found.

    TESTS = {"<segment name>": "<file from the repo root>::<test function>"}
"""

TESTS: dict = {}
