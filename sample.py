#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    from porous.sampling.config import parse_config
    from porous.sampling.run import main

    main(parse_config())
