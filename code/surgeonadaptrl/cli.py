from __future__ import annotations

import argparse
import logging

from .configuration import ExperimentConfig
from .policy import SurgeonPolicy
from .runtime import finalize_distributed, initialize_distributed
from .training import BasePolicyTrainer, make_loader


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="surgeonadaptrl-train")
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--seed", type=int, default=13)
    return command


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = ExperimentConfig.from_yaml(arguments.config)
    context = initialize_distributed()
    try:
        policy = SurgeonPolicy(config.policy, config.data.phases)
        loader = make_loader(arguments.manifest, config, context)
        BasePolicyTrainer(policy, config, context).train(loader, arguments.seed)
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
