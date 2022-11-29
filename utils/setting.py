import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np

from utils.utils import TrainerPathConst
from trainer_configs import BaseTrainerState


def setup_experiment_identifier_from_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    """
    Summary:
        実験の各種設定を返す関数

    Args:
        args: Arguments.
        exp_type: Experiment type.

    Returns:
        group, name, config file.
    """
    if args.config_file is None:
        exp_group = args.exp_group
        exp_name = args.exp_name
        exp_type = args.exp_type
        config_file = setup_default_yaml_file(exp_name, config_dir=args.config_dir)
    
    else:
        exp_group = args.exp_group
        exp_name = ".".join(str(Path(args.config_file).parts[-1]).split(".")[:-1])
        exp_type = args.exp_type
        config_file = args.config_file
    
    print(f"Source config: {config_file}")
    print(f"Results path:  {args.log_dir}/{exp_type}/{exp_group}/{exp_name}")
    
    return exp_group, exp_name, exp_type, config_file


def setup_default_yaml_file(
        exp_name: str, config_dir: str = TrainerPathConst.DIR_CONFIG) -> Path:
    """
    Summary:
        config fileが与えられなかったときに、
        defaultのyamlファイルへのpathを返す

    Args:
        config_dir: Config directory :(config)
        exp_name: Experiment name    :(default)

    Returns:
        Path to config yaml.
        config/exp_name.yaml
    """
    return Path(config_dir, f"{exp_name}.yaml")


class ExperimentFilesHandler:
    """
    Helper to handle with file locations, metrics etc.

    Args:
        model_type: Experiment type (retrieval, captioning, ...)
        exp_group: Experiment group.
        exp_name: Experiment name.
        run_name: Name of a single run.
        log_dir: Save directory for experiments.
    """

    def __init__(
            self, model_type: str, exp_group: str, exp_name: str, run_name: str, *,
            log_dir: str = TrainerPathConst.DIR_EXPERIMENTS):
        self.exp_group: str = exp_group
        self.exp_name: str = exp_name
        self.run_name: str = run_name
        self.model_type: str = model_type
        self.path_base: Path = Path(log_dir, "{}_{}".format(self.exp_name, self.run_name))
        self.path_logs = self.path_base / TrainerPathConst.DIR_LOGS
        self.path_models = self.path_base / TrainerPathConst.DIR_MODELS
        self.path_metrics = self.path_base / TrainerPathConst.DIR_METRICS
        self.path_tensorb = self.path_base / TrainerPathConst.DIR_TB
        self.path_embeddings = self.path_base / TrainerPathConst.DIR_EMBEDDINGS

    def setup_dirs(self, *, reset: bool = False) -> None:
        """
        Make sure all directories exist, delete them if a reset is requested.

        Args:
            reset: Delete this experiment.
        """
        if reset:
            # delete base path
            shutil.rmtree(self.path_base, ignore_errors=True)
            time.sleep(0.1)  # this avoids "cannot create dir that exists" on windows

        # create all paths
        for path in self.path_logs, self.path_models, self.path_metrics, self.path_tensorb:
            os.makedirs(path, exist_ok=True)

    def get_existing_checkpoints(self) -> List[int]:
        """
        Get list of all existing checkpoint numbers..

        Returns:
            List of checkpoint numbers.
        """
        # get list of existing trainerstate filenames
        list_of_files = glob.glob(str(self.get_trainerstate_file("*")))

        # extract epoch numbers from those filenames
        ep_nums = sorted([int(a.split(f"{TrainerPathConst.FILE_PREFIX_TRAINERSTATE}_")[-1].split(".json")[0])
                          for a in list_of_files])
        return ep_nums

    def find_best_epoch(self):
        """
        Find best episode out of existing checkpoint data.

        Returns:
            Best epoch or -1 if no epochs are found.
        """
        ep_nums = self.get_existing_checkpoints()
        if len(ep_nums) == 0:
            # no checkpoints found
            return -1

        # read trainerstate of the last epoch (contains all info needed to find the best epoch)
        temp_state = BaseTrainerState.create_from_file(self.get_trainerstate_file(ep_nums[-1]))
        if len(temp_state.infos_val_epochs) == 0:
            # no validation has been done, assume last epoch is best
            return ep_nums[-1]

        # read the flags for each epoch that state whether that was a good or bad epoch
        # the last good epoch is the best one
        where_res = np.where(temp_state.infos_val_is_good)[0]
        best_idx = where_res[-1]
        best_epoch = temp_state.infos_val_epochs[best_idx]
        return best_epoch

    def find_last_epoch(self):
        """
        Find last episode out of existing checkpoint data.

        Returns:
            Last epoch or -1 if no epochs are found.
        """
        ep_nums = self.get_existing_checkpoints()
        if len(ep_nums) == 0:
            # no checkpoints found
            return -1
        # return last epoch
        return ep_nums[-1]

    def get_existing_metrics(self) -> List[int]:
        """
        Get list checkpoint numbers by epoch metrics.

        Returns:
            List of checkpoint numbers.
        """
        # get list of existing trainerstate filenames
        list_of_files = glob.glob(str(self.get_metrics_epoch_file("*")))

        # extract epoch numbers from those filenames
        ep_nums = sorted([int(a.split(f"{TrainerPathConst.FILE_PREFIX_METRICS_EPOCH}_")[-1].split(".json")[0])
                          for a in list_of_files])
        return ep_nums

    # ---------- File definitions. ----------

    # Parameter epoch allows str to create glob filenames with "*".

    def get_models_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_MODEL}_{epoch}.pth"

    def get_models_file_ema(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model EMA weights.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_MODELEMA}_{epoch}.pth"

    def get_optimizer_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_OPTIMIZER}_{epoch}.pth"

    def get_data_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the optimizer.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_DATA}_{epoch}.pth"

    def get_trainerstate_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the state of the trainer. This is needed for currectly resuming training.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_TRAINERSTATE}_{epoch}.json"

    def get_metrics_step_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing step-based metrics.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_metrics / f"{TrainerPathConst.FILE_PREFIX_METRICS_STEP}_{epoch}.json"

    def get_metrics_epoch_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing epoch-based metrics.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_metrics / f"{TrainerPathConst.FILE_PREFIX_METRICS_EPOCH}_{epoch}.json"

    def get_profile_file(self) -> Path:
        """
        Get file path for storing profiling results.

        Returns:
            Path.
        """
        profile_dir = Path("profiles") / self.exp_group
        pro_file = profile_dir / (self.exp_name + ".json")
        if pro_file.is_file():
            return json.load(pro_file.open("rt", encoding="utf8"))
        return None
