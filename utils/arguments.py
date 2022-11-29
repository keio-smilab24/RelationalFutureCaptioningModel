import argparse
import logging
from typing import Any, Dict, List

# TODO :
from utils.utils import TrainerPathConst


def parse_with_config(parser: argparse.ArgumentParser):
    pass


def set_parser():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # set log option
    parser = add_default_args(parser)
    # set exp setting
    parser = add_exp_setting(parser)
    # set exp running setting (trainer)
    parser = add_trainer_setting(parser)
    # feature preloading
    parser.add_argument("--preload", action="store_true", help="Preload everything.")
    # set mart setting
    parser = add_mart_setting(parser)
    # set others setting
    parser.add_argument('--dpath', type=str, default="data", help="instread of repo_config.py")
    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--print_model", action="store_true", help=f"Print model")
    parser.add_argument('--datatype', type=str, default="bila", choices=['bila', 'bilas'])
    parser.add_argument('--wandb', '-w', action="store_true")
    parser.add_argument('--show_log', '-l', action="store_true")
    parser.add_argument('--del_weights', '-dw', action="store_true")

    return parser.parse_args()


def add_default_args(parser: argparse.ArgumentParser) -> None:
    """
    Set logging options -q, -v and test flag -t
    """
    group = parser.add_mutually_exclusive_group()
    group.set_defaults(log_level=logging.INFO)
    group.add_argument(
        "-v", "--verbose", help="Verbose (debug) logging",
        action="store_const", const=logging.DEBUG, dest="log_level")
    group.add_argument(
        "-q", "--quiet", help="Silent mode, only log warnings",
        action="store_const", const=logging.WARN, dest="log_level")
    group.add_argument(
        "--log", help="Set log level manually", type=str, dest="log_level")
    
    return parser


def add_exp_setting(parser: argparse.ArgumentParser) -> None:
    """
    Set exp setting to parser
    """
    parser.add_argument("-c", "--config_file", type=str, default=None,
                        help="Specify either config file location or experiment group and name.")
    parser.add_argument("-g", "--exp_group", type=str, default="default",
                        help="Experiment group. Path to config: config/$TYPE/$GROUP/$NAME.yaml")
    
    parser.add_argument("-e", "--exp_name", type=str, default="default",
                        help="Experiment name. Path to config: config/$TYPE/$GROUP/$NAME.yaml")
    parser.add_argument('-et', '--exp_type', type=str, default="caption")

    parser.add_argument("-n", "--num_runs", type=int, default=1, help="How many runs to do.")
    parser.add_argument("-a", "--start_run", type=int, default=1, help="Start at which run number.")
    parser.add_argument("-r", "--run_name", type=str, default="run",
                        help="Run name to save the model. Must not contain underscores.")
    
    return parser

def add_trainer_setting(parser: argparse.ArgumentParser):
    """
    Set exp running setting to parser
    """
    # configuration loading
    parser.add_argument("-o", "--config", type=str, default=None,
                        help="Modify the loaded YAML config. E.g. to change the number of dataloader workers "
                                "and the batchsize, use '-c dataloader.num_workers=20;train.batch_size=32'")
    parser.add_argument("--bs", "--batch_size", dest="batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-ls", "--label_smoothing", dest="label_smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--print_config", action="store_true", help="Print the experiment config.")
    
    # num workers
    parser.add_argument("--workers", type=int, default=None, help="Shortcut for setting dataloader workers.")
    
    # dataset path
    parser.add_argument("--config_dir", type=str, default=TrainerPathConst.DIR_CONFIG, help="Folder with config files.")
    parser.add_argument("--log_dir", type=str, default=TrainerPathConst.DIR_EXPERIMENTS,
                        help="Folder with experiment results.")
    parser.add_argument("--data_path", type=str, default=None, help="Change the data path.")
    # parser.add_argument("--profiling_dir", type=str, default=TrainerPathConst.DIR_PROFILING,
    #                         help="Profiling output.")
    
    # checkpoint loading
    parser.add_argument("--load_epoch", type=int, default=None, help="Load epoch number.")
    parser.add_argument("--load_best", action="store_true", help="Load best epoch.")
    
    # validation
    parser.add_argument("--validate", action="store_true", help="Validation only.")
    parser.add_argument("--ignore_untrained", action="store_true", help="Validate even if no checkpoint was loaded.")
    
    # reset (delete everything)
    parser.add_argument("--reset", action="store_true", help="Delete experiment.")
    parser.add_argument("--print_graph", action="store_true", help="Print model and forward pass, then exit.")
    parser.add_argument("--seed", type=str, default=None,
                        help="Set seed. integer or none/null for auto-generated seed.")
    
    # GPU
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--single_gpu", action="store_true", help="Disable multi GPU with nn.DataParallel.")

    return parser

def add_mart_setting(parser: argparse.ArgumentParser) -> None:
    """
    Add some additional arguments that are required for mart.

    Args:
        parser: Command line argument parser.
    """
    # paths
    parser.add_argument(
        "--cache_dir", type=str, default="cache_caption", help="Cached vocabulary dir."
    )
    parser.add_argument(
        "--coot_feat_dir",
        type=str,
        default="provided_embeddings",
        help="COOT Embeddings dir.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Annotations dir."
    )
    parser.add_argument(
        "--video_feature_dir",
        type=str,
        default="data/mart_video_feature",
        help="Dir containing the video features",
    )

    # Technical
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument(
        "--dataset_max", type=int, default=None, help="Reduce dataset size for testing."
    )

    return parser

def add_mart_args(parser: argparse.ArgumentParser) -> None:
    """
    Add some additional arguments that are required for mart.

    Args:
        parser: Command line argument parser.
    """
    # paths
    parser.add_argument(
        "--cache_dir", type=str, default="cache_caption", help="Cached vocabulary dir."
    )
    parser.add_argument(
        "--coot_feat_dir",
        type=str,
        default="provided_embeddings",
        help="COOT Embeddings dir.",
    )
    parser.add_argument(
        "--annotations_dir", type=str, default="annotations", help="Annotations dir."
    )
    parser.add_argument(
        "--video_feature_dir",
        type=str,
        default="data/mart_video_feature",
        help="Dir containing the video features",
    )

    # Technical
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument(
        "--dataset_max", type=int, default=None, help="Reduce dataset size for testing."
    )

    return parser


def update_config_from_args(config: Dict, args: argparse.Namespace, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Modify config and paths given script arguments.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Updated config dict.
    """
    # parse the --config inline modifier
    if args.config is not None:
        # get all fields to update from the argument and loop them
        update_fields: List[str] = args.config.split(",")
        for field_value in update_fields:
            # get field and value
            fields_str, value = field_value.strip().split("=")
            # convert value if necessary
            try:
                value = float(value)
                if round(value) == value:
                    value = int(value)
            except ValueError:
                pass
            if str(value).lower() == "true":
                value = True
            elif str(value).lower() == "false":
                value = False
            # update the correct nested dictionary field
            fields = fields_str.split(".")
            current_dict = config
            for i, field in enumerate(fields):
                if i == len(fields) - 1:
                    # update field
                    if field not in current_dict:
                        assert "same_as" in current_dict, (
                            f"Field {fields_str} not found in config {list(current_dict.keys())}. "
                            f"Typo or field missing in config.")
                    current_dict[field] = value
                    if verbose:
                        print(f"    Change config: Set {fields_str} = {value}")
                    break
                # go one nesting level deeper
                current_dict = current_dict[field]

    if args.workers is not None:
        config["dataset_train"]["num_workers"] = int(args.workers)
        config["dataset_val"]["num_workers"] = int(args.workers)
        if verbose:
            print(f"    Change config: Set dataloader workers to {args.workers} for train and val.")

    if args.seed is not None:
        if str(args.seed).lower() in ["none", "null"]:
            config["random_seed"] = None
        else:
            config["random_seed"] = int(args.seed)
        if verbose:
            print(f"    Change config: Set seed to {args.seed}. Deterministic")

    if args.no_cuda:
        config["use_cuda"] = False
        if verbose:
            print(f"    Change config: Set use_cuda to False.")

    if args.single_gpu:
        config["use_multi_gpu"] = False
        if verbose:
            print(f"    Change config: Set use_multi_gpu to False.")

    return config


def update_mart_config_from_args(
    config: Dict, args: argparse.Namespace, *, verbose: bool = True
) -> Dict[str, Any]:
    """
    Modify config given script arguments.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Updated config dict.
    """
    if args.debug:
        config["debug"] = True
        if verbose:
            print(f"    Change config: Set debug to True")
    if args.dataset_max is not None:
        assert args.dataset_max > 0, "--dataset_max must be positive int."
        config["dataset_train"]["max_datapoints"] = args.dataset_max
        config["dataset_val"]["max_datapoints"] = args.dataset_max
        if verbose:
            print(
                f"    Change config: Set dataset_(train|val).max_datapoints to {args.dataset_max}"
            )
    config["train"]["batch_size"] = args.batch_size
    config["label_smoothing"] = args.label_smoothing
    if args.preload:
        config["dataset_train"]["preload"] = True
        config["dataset_val"]["preload"] = True
        if verbose:
            print(f"    Change config: Set dataset_(train|val).preload to True")
    if args.no_preload or args.validate:
        config["dataset_train"]["preload"] = False
        config["dataset_val"]["preload"] = False
        if verbose:
            print(
                f"    Change config: Set dataset_(train|val).preload to False (--no_preload or --validate)"
            )
    return config
