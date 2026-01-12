import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader
from src.DDPM.guided_diffusion.VinDrMammo_dataset import *
from src.config import METADATA_ROOT

def load_data(
    *,
    data_dir,
    batch_size,
    deterministic=False,
    category="all", # healthy, anomalous or all
    mode="all", # train, test, val or all
    single_batch=False # If True, retrieve only 1 batch
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    df_path = os.path.join(METADATA_ROOT, "grouped_df.csv")

    dataset = VinDrMammoDataset(
        dataset_root_folder_filepath=data_dir,
        df_path=df_path,
        category=category,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True # Try with 1 worker
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    if single_batch:
        for batch in loader:
            yield batch
            break
    else:
        while True:
            yield from loader







