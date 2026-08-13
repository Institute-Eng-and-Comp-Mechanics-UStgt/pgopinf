import logging

from pgopinf.logging_config import setup_logging
from pgopinf.specs.presets.run_recipes import RunConfig, get_recipe
from pgopinf.workflows.run import run

DEFAULT_CONFIG = get_recipe("msd_siso_50mass_petrov_vs_galerkin")


def main(config: RunConfig = DEFAULT_CONFIG):
    run(config)
    print("Done.")


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    main()
