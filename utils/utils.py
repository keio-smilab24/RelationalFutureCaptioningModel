import random
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from nntrainer.utils import TrainerPathConst


def fix_seed(seed: int):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)


def get_reference_files(
    dset_name: str,
    annotations_dir: Union[str, Path] = TrainerPathConst.DIR_ANNOTATIONS,
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
    # TODO : bilas / bila
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
