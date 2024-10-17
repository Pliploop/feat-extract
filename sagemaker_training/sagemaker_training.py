import logging
from dataclasses import dataclass, field, asdict

from sagemaker import TrainingInput
from sagemaker.estimator import Estimator

logger = logging.getLogger(__name__)

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

@dataclass
class TrainingInputConfig:
    s3_data: str
    s3_data_type: str = "S3Prefix"
    distribution: str = "FullyReplicated"
    input_mode: str = "File"


@dataclass
class HyperParameters:
    entrypoint: str  # don't put leading ./ here
    cfg_fp: str  # don't put leading ./ here
    s3_input_location: str  # this will be dynamically determined by launch_sagemaker_training
    s3_output_location: str  # this will be dynamically determined by launch_sagemaker_training
    job_name: str  # this will be dynamically determined by launch_sagemaker_training
    input_data_dir: str = "/opt/ml/input/data/training"  # where to find the input data locally, fixed by AWS
    local: bool = False  # if True, run a local sagemaker instance
    additional_parameters: dict = field(default_factory=dict)


@dataclass
class EstimatorConfig:
    base_job_name: str
    output_path: str
    image_uri: str
    instance_count: int
    instance_type: str
    hyperparameters: dict
    tags: list[dict]
    volume_size: int = 100  # in GB, ignored with some instance_types because they come with predetermined size
    max_run: int = 5 * 24 * 60 * 60  # job timeout in seconds, corresponds to 5 days, the maximum available
    role: str = ""
    enable_sagemaker_metrics: bool = True

    def __post_init__(self):
        if isinstance(self.hyperparameters, dict):
            self.hyperparameters = HyperParameters(**self.hyperparameters)


def launch_sagemaker_training(cfg: dict):
    if cfg["estimator"]["hyperparameters"].get("local", False):
        raise NotImplementedError("Sagemaker local instances are not supported yet")

    # INFER LAST CONFIGURATION DETAILS FROM THE INPUT
    base_job_name = "-".join([cfg["base_name"], "training"])
    job_name = "-".join([base_job_name, get_timestamp()])
    output_path = f"{cfg['s3_output_root']}{cfg['base_name']}/models/{job_name}"
    cfg["estimator"]["hyperparameters"]["s3_output_location"] = output_path
    cfg["estimator"]["hyperparameters"]["s3_input_location"] = cfg["training_input"]["s3_data"]
    cfg["estimator"]["hyperparameters"]["job_name"] = job_name

    # CONFIGURING THE INPUTS
    training_input_config = TrainingInputConfig(**cfg["training_input"])
    inputs = TrainingInput(**asdict(training_input_config))

    # CONFIGURING THE ESTIMATOR
    estimator_config = EstimatorConfig(
        base_job_name=base_job_name,
        output_path=output_path,
        **cfg["estimator"],
    )
    
    
    
    
    
    estimator = Estimator(**asdict(estimator_config))

    # RUN!
    logging.info("Start the training!")
    estimator.fit(inputs=inputs, logs="All", job_name=job_name)
    
    
    