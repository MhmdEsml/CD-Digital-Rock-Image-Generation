#!/usr/bin/env python
import os

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false").strip()
os.environ.setdefault('NCCL_DEBUG', 'WARN')

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from porous.cd.run import main

if __name__ == '__main__':
    sys.exit(main())
