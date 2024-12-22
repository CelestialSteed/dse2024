import os
import sys

from algo.utils import get_configs, parse_args

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    print(configs)
