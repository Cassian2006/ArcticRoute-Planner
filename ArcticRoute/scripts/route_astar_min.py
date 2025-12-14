#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal A* routing CLI over a single NetCDF variable for debugging.

@role: core
"""

"""兼容旧入口，内部转发到 api.cli plan 子命令。"""

import sys

from ..api import cli


def main() -> int:
    argv = ["plan"] + sys.argv[1:]
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
