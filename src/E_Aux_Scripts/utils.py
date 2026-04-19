import torch
import random
import numpy as np
import socket

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def seed_worker(seed_or_args, worker_id):
	base_seed = seed_or_args.seed if hasattr(seed_or_args, "seed") else int(seed_or_args)
	worker_seed = base_seed + worker_id
	np.random.seed(worker_seed)
	random.seed(worker_seed)

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
	"""Check if internet connection is available"""
	try:
		socket.setdefaulttimeout(timeout)
		socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
		return True
	except socket.error:
		return False