# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import signal
import sys

from .cli import main

if __name__ == "__main__":
    # Killed plainly, the workers and the TP group's ranks would outlive it.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    sys.exit(main())
