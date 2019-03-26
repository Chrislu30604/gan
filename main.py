#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This is main file entry that execute handle the parser and arguments
"""

import argparse

from utils.config import *
from agents import *

__author__ = "Chrislu"
__license__ = "mit"


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line paramnter as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line paramters namepsace
    """
    parser = argparse.ArgumentParser(
        description="GAN"
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help="load the configuration json file"
    )
    return parser.parse_args(args)


def main(args):
    """Main entry point allowing external calls
    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)

    # parse the config json file and return the configuration
    config = process_config(args.config, args.loglevel)

    # Create Agent obj by config.agent and pass config
    agent_class = eval(config.agent)
    agent = agent_class(config)
    agent.run()
    agent.finalize()


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
