import argparse
import datetime
import logging
import random
import os
import sys
from pathlib import Path
from copy import deepcopy
from json import JSONEncoder
from typing import Any, Dict, List, Optional, Union, Tuple
import ctypes
import multiprocessing

import numpy as np
import torch
import GPUtil
import psutil

from utils import baseconfig
from utils.baseconfig import ConstantHolder

DEFAULT = "default"
REF = "ref"
NONE = "none"
LOGGER_NAME = "trainlog"
LOGGING_FORMATTER = logging.Formatter("%(levelname)5s %(message)s", datefmt="%m%d %H%M%S")

# ---------- Multiprocessing ----------

MAP_TYPES: Dict[str, Any] = {
    'int': ctypes.c_int,
    'long': ctypes.c_long,
    'float': ctypes.c_float,
    'double': ctypes.c_double
}

class LogLevelsConst(ConstantHolder):
    """
    Loglevels, same as logging module.
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def create_logger_without_file(name: str, log_level: int = LogLevelsConst.INFO, no_parent: bool = False,
                                no_print: bool = False) -> logging.Logger:
    """
    Create a stdout only logger.

    Args:
        name: Name of the logger.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.
    Returns:
        Created logger.
    """
    return create_logger(name, log_dir="", log_level=log_level, no_parent=no_parent, no_print=no_print)


def create_logger(
        name: str, *, filename: str = "run", log_dir: Union[str, Path] = "", log_level: int = LogLevelsConst.INFO,
        no_parent: bool = False, no_print: bool = False) -> logging.Logger:
    """
    Create a new logger.

    Notes:
        This created stdlib logger can later be retrieved with logging.getLogger(name) with the same name.
        There is no need to pass the logger instance between objects.

    Args:
        name: Name of the logger.
        log_dir: Target logging directory. Empty string will not create files.
        filename: Target filename.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.

    Returns:
    """
    # create logger, set level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # remove old handlers to avoid duplicate messages
    remove_handlers_from_logger(logger)

    # file handler
    file_path = None
    if log_dir != "":
        ts = get_timestamp_for_filename()
        file_path = Path(log_dir) / "{}_{}.log".format(filename, ts)
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(LOGGING_FORMATTER)
        logger.addHandler(file_hdlr)

    # stdout handler
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(strm_hdlr)

    # disable propagating to parent to avoid double logs
    if no_parent:
        logger.propagate = False

    if not no_print:
        print(f"Logger: '{name}' to {file_path}")
    return logger


def remove_handlers_from_logger(logger: logging.Logger) -> None:
    """
    Remove handlers from the logger.

    Args:
        logger: Logger.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def print_logger_info(logger: logging.Logger) -> None:
    """
    Print infos describing the logger: The name and handlers.

    Args:
        logger: Logger.
    """
    print(logger.name)
    x = list(logger.handlers)
    for i in x:
        handler_str = f"Handler {i.name} Type {type(i)}"
        print(handler_str)


# ---------- Argparser ----------


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter,
                        argparse.MetavarTypeHelpFormatter):
    """
    Custom formatter
    - raw descriptions (no removing newlines)
    - show default values
    - show metavars (str, int, ...) instead of names
    - fit to console width
    """

    def __init__(self, prog: Any):
        try:
            term_size = os.get_terminal_size().columns
            max_help_pos = term_size // 2
        except OSError:
            term_size = None
            max_help_pos = 24
        super().__init__(
            prog, max_help_position=max_help_pos, width=term_size)


class ArgParser(argparse.ArgumentParser):
    """
    ArgumentParser with Custom Formatter and some convenience functions.

    For best results, write a docstring at the top of the file and call it
    with ArgParser(description=__doc__)

    Args:
        description: Help text for Argparser. Set description=__doc__ and write help text into module header.
    """

    def __init__(self, description: str = "none"):
        super().__init__(description=description, formatter_class=CustomFormatter)


# ---------- Time utilities ----------

def get_timestamp_for_filename(dtime: Optional[datetime.datetime] = None):
    """
    Convert datetime to timestamp for filenames.

    Args:
        dtime: Optional datetime object, will use now() if not given.

    Returns:
    """
    if dtime is None:
        dtime = datetime.datetime.now()
    ts = str(dtime).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


# ---------- Files ----------
def parse_file_to_list(file: Union[Path, str]) -> List[str]:
    """
    Given a text file, read contents to list of lines. Strip lines, ignore empty and comment lines

    Args:
        file: Input file.

    Returns:
        List of lines.
    """
    # loop lines
    output = []
    for line in Path(file).read_text(encoding="utf8").splitlines(keepends=False):
        # strip line
        line = line.strip()
        if line == "":
            # skip empty line
            continue
        if line[0] == "#":
            # skip comment line
            continue
        # collect
        output.append(line)
    return output


# ---------- Config / dict ----------

def resolve_sameas_config_recursively(config: Dict, *, root_config: Optional[Dict] = None):
    """
    Recursively resolve config fields described with same_as.

    If any container in the config has the field "same_as" set, find the source identifier and copy all data
    from there to the target container. The source identifier can nest with dots e.g.
    same_as: "net_video_local.input_fc_config" will copy the values from container input_fc_config located inside
    the net_video_local container.

    Args:
        config: Config to modify.
        root_config: Config to get the values from, usually the same as config.

    Returns:
    """
    if root_config is None:
        root_config = config
    # loop the current config and check
    loop_keys = list(config.keys())
    for key in loop_keys:
        value = config[key]
        if not isinstance(value, dict):
            continue
        same_as = value.get("same_as")
        if same_as is not None:
            # current container should be filled with the values from the source container. loop source container
            source_container = get_dict_value_recursively(root_config, same_as)
            for key_source, val_source in source_container.items():
                # only write fields that don't exist yet, don't overwrite everything
                if key_source not in config[key]:
                    # at this point we want a deepcopy to make sure everything is it's own object
                    config[key][key_source] = deepcopy(val_source)
            # at this point, remove the same_as field.
            del value["same_as"]

        # check recursively
        resolve_sameas_config_recursively(config[key], root_config=root_config)


def get_dict_value_recursively(dct: Dict, key: str) -> Any:
    """
    Nest into the dict given a key like root.container.subcontainer

    Args:
        dct: Dict to get the value from.
        key: Key that can describe several nesting steps at one.

    Returns:
        Value.
    """
    key_parts = key.split(".")
    if len(key_parts) == 1:
        # we arrived at the leaf of the dict tree and can return the value
        return dct[key_parts[0]]
    # nest one level deeper
    return get_dict_value_recursively(dct[key_parts[0]], ".".join(key_parts[1:]))


def check_config_dict(name: str, config: Dict[str, Any], strict: bool = True) -> None:
    """
    Make sure config has been read correctly with .pop(), and no fields are left over.

    Args:
        name: config name
        config: config dict
        strict: Throw errors
    """
    remaining_keys, remaining_values = [], []
    for key, value in config.items():
        if key == REF:
            # ignore the reference configurations, they can later be used for copying things with same_as
            continue
        remaining_keys.append(key)
        remaining_values.append(value)
    # check if something is left over
    if len(remaining_keys) > 0:
        if not all(value is None for value in remaining_values):
            err_msg = (
                f"keys and values remaining in config {name}: {remaining_keys}, {remaining_values}. "
                f"Possible sources of this error: Typo in the field name in the yaml config file. "
                f"Incorrect fields given with --config flag. "
                f"Field should be added to the config class so it can be parsed. "
                f"Using 'same_as' and forgot to set these fields to null.")

            if strict:
                print(f"Print config for debugging: {config}")
                raise ValueError(err_msg)
            logging.getLogger(LOGGER_NAME).warning(err_msg)

class BetterJSONEncoder(JSONEncoder):
    """
    Enable the JSON encoder to handle Path objects.

    It would be nice to also handle numpy arrays, tensors etc. but that is not required currently.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


# ---------- Constants ----------

class ConfigNamesConst(baseconfig.ConstantHolder):
    """
    Stores configuration group names.
    """
    TRAIN = "train"
    VAL = "val"
    DATASET_TRAIN = "dataset_train"
    DATASET_VAL = "dataset_val"
    LOGGING = "logging"
    SAVING = "saving"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


class TrainerPathConst():
    """
    Stores directory and file names for training.
    """
    DIR_CONFIG = "config"
    DIR_EXPERIMENTS = "results"
    DIR_LOGS = "logs"
    DIR_MODELS = "models" 
    DIR_METRICS = "metrics"
    DIR_EMBEDDINGS = "embeddings"
    DIR_TB = "tb"
    DIR_CAPTION = "caption"
    DIR_ANNOTATIONS = "data" 
    FILE_PREFIX_TRAINERSTATE = "trainerstate"
    FILE_PREFIX_MODEL = "model"
    FILE_PREFIX_MODELEMA = "modelema"
    FILE_PREFIX_OPTIMIZER = "optimizer"
    FILE_PREFIX_DATA = "data"
    FILE_PREFIX_METRICS_STEP = "metrics_step"
    FILE_PREFIX_METRICS_EPOCH = "metrics_epoch"
    FILE_PREFIX_TRANSL_RAW = "translations"


class MetricComparisonConst(baseconfig.ConstantHolder):
    """
    Fields for the early stopper.
    """
    # metric comparison
    VAL_DET_BEST_MODE_MIN = "min"
    VAL_DET_BEST_MODE_MAX = "max"
    VAL_DET_BEST_TH_MODE_REL = "rel"
    VAL_DET_BEST_TH_MODE_ABS = "abs"


def fix_seed(seed: int, cudnn_deterministic: bool=False, cudnn_benchmark: bool=True):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    # torch.use_deterministic_algorithms(True)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)


def get_reference_files(
    dset_name: str,
    annotations_dir: Union[str, Path],
    test: bool = False,
    datatype: str = 'bila',
) -> Dict[str, List[Path]]:
    """
    Given dataset name, load the ground truth annotations for captioning.

    Args:
        dset_name: Dataset name.
        annotations_dir: Folder with annotations.

    Returns:
        Dictionary with key: evaluation_mode (val, test) and value: list of annotation files.
    """
    annotations_dir = Path(annotations_dir) / dset_name
    if dset_name == "activitynet":
        return {
            "val": [
                annotations_dir / "captioning_val_1_para.json",
                annotations_dir / "captioning_val_2_para.json",
            ],
            "test": [
                annotations_dir / "captioning_test_1_para.json",
                annotations_dir / "captioning_test_2_para.json",
            ],
        }
    if dset_name == "youcook2":
        return {"val": [annotations_dir / "captioning_val_para.json"]}
    if dset_name == "ponnet":
        if test:
            return {"test": [annotations_dir / "captioning_test_para.json"]}
        else:
            return {"val": [annotations_dir / "captioning_val_para.json"]}
    if dset_name == "BILA":
        if datatype == 'bila':
            if test:
                return {"test": [annotations_dir / "captioning_test_para.json"]}
            else:
                return {"val": [annotations_dir / "captioning_val_para.json"]}
        if datatype == 'bilas':
            if test:
                return {"test": ["data/BilaS/caption_test.json"]}
            else:
                return {"val": ["data/BilaS/caption_valid.json"]}
    raise ValueError(f"Dataset unknown {dset_name}")


def create_shared_array(arr: np.ndarray, dtype: str = "float") -> np.array:
    """
    Converts an existing numpy array into a shared numpy array, such that
    this array can be used by multiple CPUs. Used e.g. for preloading the
    entire feature dataset into memory and then making it available to multiple
    dataloaders.

    Args:
        arr (np.ndarray): Array to be converted to shared array
        dtype (np.dtype): Datatype of shared array

    Returns:
        shared_array (multiprocessing.Array): shared array
    """
    shape = arr.shape
    flat_shape = int(np.prod(np.array(shape)))
    c_type = MAP_TYPES[dtype]
    shared_array_base = multiprocessing.Array(c_type, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    shared_array[:] = arr[:]
    return shared_array


def get_truncnorm_tensor(shape: Tuple[int], *, mean: float = 0, std: float = 1, limit: float = 2) -> torch.Tensor:
    """
    Create and return normally distributed tensor, except values with too much deviation are discarded.

    Args:
        shape: tensor shape
        mean: normal mean
        std: normal std
        limit: which values to discard

    Returns:
        Filled tensor with shape (*shape)
    """
    assert isinstance(shape, (tuple, list)), f"shape {shape} is not a tuple or list of ints"
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def fill_tensor_with_truncnorm(input_tensor: torch.Tensor, *, mean: float = 0, std: float = 1, limit: float = 2) -> None:
    """
    Fill given input tensor with a truncated normal dist.

    Args:
        input_tensor: tensor to be filled
        mean: normal mean
        std: normal std
        limit: which values to discard
    """
    # get truncnorm values
    tmp = get_truncnorm_tensor(input_tensor.shape, mean=mean, std=std, limit=limit)
    # fill input tensor
    input_tensor[...] = tmp[...]


# ---------- Profiling ----------

def profile_gpu_and_ram() -> Tuple[List[str], List[float], List[float], List[float], float, float, float]:
    """
    Profile GPU and RAM.

    Returns:
        GPU names, total / used memory per GPU, load per GPU, total / used / available RAM.
    """

    # get info from gputil
    _str, dct_ = _get_gputil_info()
    dev_num = os.getenv("CUDA_VISIBLE_DEVICES")
    if dev_num is not None:
        # single GPU set with OS flag
        gpu_info = [dct_[int(dev_num)]]
    else:
        # possibly multiple gpus, aggregate values
        gpu_info = []
        for dev_dict in dct_:
            gpu_info.append(dev_dict)

    # convert to GPU info and MB to GB
    gpu_names: List[str] = [gpu["name"] for gpu in gpu_info]
    total_memory_per: List[float] = [gpu["memoryTotal"] / 1024 for gpu in gpu_info]
    used_memory_per: List[float] = [gpu["memoryUsed"] / 1024 for gpu in gpu_info]
    load_per: List[float] = [gpu["load"] / 100 for gpu in gpu_info]

    # get RAM info and convert to GB
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    ram_avail: float = mem.available / 1024 ** 3

    return gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail


def _get_gputil_info():
    """
    Returns info string for printing and list with gpu infos. Better formatting than the original GPUtil.

    Returns:
        gpu info string, List[Dict()] of values. dict example:
            ('id', 1),
            ('name', 'GeForce GTX TITAN X'),
            ('temperature', 41.0),
            ('load', 0.0),
            ('memoryUtil', 0.10645266950540452),
            ('memoryTotal', 12212.0)])]
    """

    gpus = GPUtil.getGPUs()
    attr_list = [
        {'attr': 'id', 'name': 'ID'}, {'attr': 'name', 'name': 'Name'},
        {'attr': 'temperature', 'name': 'Temp', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
        {'attr': 'load', 'name': 'GPU util.', 'suffix': '% GPU', 'transform': lambda x: x * 100,
         'precision': 1},
        {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '% MEM', 'transform': lambda x: x * 100,
         'precision': 1}, {'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
        {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0}
    ]
    gpu_strings = [''] * len(gpus)
    gpu_info = []
    for _ in range(len(gpus)):
        gpu_info.append({})

    for attrDict in attr_list:
        attr_precision = '.' + str(attrDict['precision']) if (
                'precision' in attrDict.keys()) else ''
        attr_suffix = str(attrDict['suffix']) if (
                'suffix' in attrDict.keys()) else ''
        attr_transform = attrDict['transform'] if (
                'transform' in attrDict.keys()) else lambda x: x
        for gpu in gpus:
            attr = getattr(gpu, attrDict['attr'])

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = '{0:d}'.format(attr)
            elif isinstance(attr, str):
                attr_str = attr
            else:
                raise TypeError('Unhandled object type (' + str(
                    type(attr)) + ') for attribute \'' + attrDict[
                                    'name'] + '\'')

            attr_str += attr_suffix

        for gpuIdx, gpu in enumerate(gpus):
            attr_name = attrDict['attr']
            attr = getattr(gpu, attr_name)

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = ('{0:' + 'd}').format(attr)
            elif isinstance(attr, str):
                attr_str = ('{0:' + 's}').format(attr)
            else:
                raise TypeError(
                    'Unhandled object type (' + str(
                        type(attr)) + ') for attribute \'' + attrDict[
                        'name'] + '\'')
            attr_str += attr_suffix
            gpu_info[gpuIdx][attr_name] = attr
            gpu_strings[gpuIdx] += '| ' + attr_str + ' '

    return "\n".join(gpu_strings), gpu_info


def count_parameters(model, verbose=True):
    """
    Summary:
        model内の学習可能なパラメータ・固定パラメータの数を返す
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"Parameters total: {n_all}, frozen: {n_frozen}")
    return n_all, n_frozen