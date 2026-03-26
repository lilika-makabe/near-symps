import inspect
import os
from logging import getLogger, config


def parse_logger(output_logdir, output_prefix=None, name="__main__",
                 log_conf_path=os.path.join(os.path.dirname(__file__), "configs", "log_config.json"), sys_io=True,
                 file_io=True):
    '''
    return logger.
    # ex.
    logger = parse_logger(name=__name__, output_logdir="./log")
    # else is same as always:
    logger.info("logger is created.")

    :param name: name passed to getLogger().
    :param output_prefix: default is {filename}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log
    :param output_logdir: output log dir.
    :param log_conf_path: set json path or default setting will be used.
    :return: logger
    '''
    from datetime import datetime, timedelta, timezone
    JST = timezone(timedelta(hours=+9), 'JST')
    time = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_logdir, exist_ok=True)

    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
    log_conf = read_json_config(log_conf_path)
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(output_logdir, f"{output_prefix}_{time}.log")
    config.dictConfig(log_conf)
    logger = getLogger(name)
    print(os.path.join(output_logdir, f"{output_prefix}_{time}.log"))
    return logger


def read_json_config(log_conf_path: str):
    import json
    with open(log_conf_path, 'r') as f:
        log_conf = json.load(f)
    return log_conf
