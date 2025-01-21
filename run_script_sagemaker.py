
import logging

from sagemaker_training.sagemaker_processing import launch_sagemaker_processing
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


import argparse

if __name__ == "__main__":
    # Set up argument parsing to extract --config from the command line
    parser = argparse.ArgumentParser(description="SageMaker Processing Launcher")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file")
    args, unknown_args = parser.parse_known_args()

    cfg_path = args.config

    # Configure logging
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    # Load the configuration from the YAML file
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path))

    # Update the entrypoint with the rest of the command-line arguments as a string
    if "entrypoint" in cfg:
        cfg["entrypoint"] +=  [" ".join(unknown_args)]

    logger.info("Launching SageMaker Processing with configuration: %s", cfg)

    # Launch the SageMaker processing job
    launch_sagemaker_processing(cfg)