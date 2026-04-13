import torch
import random
import numpy as np
import socket

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def seed_worker(args, worker_id):
	worker_seed = args.seed + worker_id
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