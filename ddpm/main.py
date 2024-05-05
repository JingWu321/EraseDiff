import argparse
import traceback
import logging
import sys
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from functions import get_config_and_setup_dirs, get_mask_config_and_setup_dirs

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    #
    parser.add_argument('--unl_ratio', default=0.1, type=float)
    parser.add_argument('--unl_cls', default='0', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--unl_eps', type=int, default=4)
    parser.add_argument('--K_steps', default=2, type=int, help='K steps for LL')
    parser.add_argument('--S_steps', default=2, type=int, help='S steps for UL')
    parser.add_argument('--lambda_bome', type=float, default=0.1)
    parser.add_argument('--eta_bome', type=float, default=0.1)
    parser.add_argument('--rem_num', type=int, default=4500)
    # SalUn
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="forget loss vs remain loss"
    )
    parser.add_argument(
        "--mask_path", type=str, default=None, help="the path to store mask"
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.5, help="mask ratio for unlearning"
    )
    parser.add_argument(
        "--method", type=str, default=None, help="the method to unlearn"
    )
    #
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--ckpt_folder", type=str, default='./results/cifar10')
    parser.add_argument(
        "--mode", type=str, default='train',
        help="['train', 'forget', 'ours', 'ft', 'ng', 'blindspot', 'saliency_unlearn']"
    )
    parser.add_argument(
        "--label_to_forget", type=int, default=0, help="Class label 0-9 to forget. Only for forgetting training."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    # parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=2.0,
        help="classifier-free guidance conditional strength"
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    # config = get_config_and_setup_dirs(os.path.join("configs", args.config))
    if args.mode == "saliency_unlearn":
        config = get_mask_config_and_setup_dirs(
            args, os.path.join("configs", args.config)
        )
    else:
        config = get_config_and_setup_dirs(os.path.join("configs", args.config))


    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    args, config = parse_args_and_config()
    if args.mode == 'train':
        logging.info("Writing log file to {}".format(config.log_dir))

    try:
        runner = Diffusion(args, config)
        if args.mode == 'ours':
            runner.train_ours()
        elif args.mode == 'forget':
            runner.train_forget()
        elif args.mode == 'ft':
            runner.train_ft()
        elif args.mode == 'ng':
            runner.train_ng()
        elif args.mode == 'blindspot':
            runner.train_blindspot()
        elif args.mode == "saliency_unlearn":
            runner.saliency_unlearn()
        elif args.mode == "generate_mask":
            runner.generate_mask()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
