import os
import time
import yaml
import argparse
from typing import Dict, Optional, NoReturn
def parse_args() -> argparse.Namespace:
    def initialize_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
        	"-c",
            "--configs",
            # required=True,
            type=str,
            default="configs.yml",
            help="YAML file to be handled"
        )

        return parser

    parser = argparse.ArgumentParser(
    	formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = initialize_parser(parser)
    return parser.parse_args()

def get_configs(fyaml: str) -> Dict:
    if_exist(fyaml, strict=True)
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs

def if_exist(path: str, strict: bool = False) -> Optional[bool]:
	if os.path.exists(path):
		return True
	else:
		warn("{} is not found.".format(path))
		if strict:
			exit(1)
		else:
			return False

def warn(msg: str) -> NoReturn:
    print("[WARN]: {}".format(msg))