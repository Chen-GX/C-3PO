import logging
import sys
import os
import json
import warnings
import os.path as osp
warnings.filterwarnings("ignore", category=DeprecationWarning) 
logger = logging.getLogger(__name__)

def config_logging(file_name, write_to_file=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    if write_to_file:
        file_handler = logging.FileHandler(file_name, mode='a', encoding="utf8")
        # %(asctime)s - [%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s] - %(message)s
        formatter = logging.Formatter(fmt, datefmt="%Y/%m/%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y/%m/%d %H:%M:%S"))
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler] if write_to_file else [console_handler],
    )


def log_params(FLAGS, write_to_file=True):
    # 配置logging
    if write_to_file:
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    config_logging(osp.join(FLAGS.output_dir, 'logfile.log'), write_to_file)

    if write_to_file:
        for k, v in FLAGS.__dict__.items():
            logger.info(k + ":" + str(v))

        with open(osp.join(FLAGS.output_dir, 'commandline_args.json'), 'w') as f:
            json.dump(FLAGS.__dict__, f, indent=2)


