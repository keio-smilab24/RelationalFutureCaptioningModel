"""
Train captioning with MART.

Originally published by https://github.com/jayleicn/recurrent-transformer under MIT license
Reworked by https://github.com/gingsi/coot-videotext under Apache 2 license
"""

import numpy as np

from coot.configs_retrieval import ExperimentTypesConst
from mart import arguments_mart
from mart.configs_mart import MartConfig as Config
from mart.model import create_mart_model
from mart.recursive_caption_dataset import create_mart_datasets_and_loaders
from mart.trainer_caption import MartTrainer
from nntrainer import arguments, utils
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file
import datetime
from utils.arguments import set_parser
from utils.utils import fix_seed


EXP_TYPE = ExperimentTypesConst.CAPTION


def main():
    args = set_parser()

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config given the script arguments
    config = arguments.update_config_from_args(config, args)
    config = arguments_mart.update_mart_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config)
    if args.print_config:
        print(cfg)

    # set seed
    verb = "Set seed"
    if cfg.random_seed is None:
        cfg.random_seed = np.random.randint(0, 2 ** 15, dtype=np.int32)
        verb = "Randomly generated seed"
    print(f"{verb} {cfg.random_seed} deterministic {cfg.cudnn_deterministic} "
            f"benchmark {cfg.cudnn_benchmark}")
    set_seed(cfg.random_seed, cudnn_deterministic=cfg.cudnn_deterministic, cudnn_benchmark=cfg.cudnn_benchmark)

    # create dataset
    train_set, val_set, train_loader, val_loader, test_set, test_loader = create_mart_datasets_and_loaders(
        cfg, args.coot_feat_dir, args.annotations_dir, args.video_feature_dir, datatype=args.datatype)

    for i in range(args.start_run):
        run_number = datetime.datetime.now()
        run_name = f"{args.run_name}{run_number}"

        model = create_mart_model(cfg, len(train_set.word2idx), cache_dir=args.cache_dir)

        if args.print_model and i == 0:
            print(model)

        # always load best epoch during validation
        load_best = args.load_best or args.validate

        # create trainer
        trainer = MartTrainer(
            cfg, model, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset, load_best=load_best,
            load_epoch=args.load_epoch, load_model=args.load_model, inference_only=args.validate,
            annotations_dir=args.annotations_dir)

        if args.validate:
            if not trainer.load and not args.ignore_untrained:
                raise ValueError("Validating an untrained model! No checkpoints were loaded. Add --ignore_untrained "
                                    "to ignore this error.")
            trainer.validate_epoch(val_loader, datatype=args.datatype)
        else:
            trainer.train_model(train_loader, val_loader, test_loader, datatype=args.datatype)

        trainer.close()
        del model
        del trainer


if __name__ == "__main__":
    main()
