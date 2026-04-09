from .util import convert_dictconfig_to_dict, seed_everything  # NOQA


def run(*args, **kwargs):
	from .trainer import run as _run
	return _run(*args, **kwargs)


def run_ddp(*args, **kwargs):
	from .trainer_ddp import run_ddp as _run_ddp
	return _run_ddp(*args, **kwargs)


def run_validation(*args, **kwargs):
	from .validator import run_validation as _run_validation
	return _run_validation(*args, **kwargs)
