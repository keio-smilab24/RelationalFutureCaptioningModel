import argparse
from pathlib import Path
from typing import Tuple

from nntrainer.utils import TrainerPathConst


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