import pandas as pd
import os
import torch
import imageio


class VinDrMammoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root_folder_filepath,
            df_path,
            transform = None,
            category = "all", # Can be "healthy"/"anomalous"/"all"

    ):
        super(VinDrMammoDataset, self).__init__()
        self.dataset_root_folder_filepath = dataset_root_folder_filepath
        self.df_path = df_path
        self.category = category
        self.transform = transform

        self.meta_data_data_frame = pd.read_csv(
            self.df_path, encoding="ISO-8859-1"
        )
        if category == "healthy":
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['breast_birads'] == 'BI-RADS 1']
            print(f"Number of healthy samples: {len(self.meta_data_data_frame)}")

        elif category == "anomalous":
            self.meta_data_data_frame = self.meta_data_data_frame[~self.meta_data_data_frame["finding_categories"].str.contains('No Finding', na=False)]
            print(f"Number of anomalous samples with findings: {len(self.meta_data_data_frame)}")

        self.sample_idx_to_scan_path_and_label = []

        self.sample_idx_to_scan_path_and_label = [
            (os.path.join(self.dataset_root_folder_filepath, row["image_id"]),
               row["breast_birads"])
            for _, row in self.meta_data_data_frame.iterrows()
        ]


    def __len__(self):

        return len(self.sample_idx_to_scan_path_and_label)

    def __getitem__(self, item):
        # Retrieve image_path and "breast_birads"
        image_path, category = self.sample_idx_to_scan_path_and_label[item]

        image = imageio.imread(image_path)
        image = torch.tensor(image, dtype=torch.float32)
        
        if self.transform is not None:
            image = self.transform(image)

        if torch.max(image) > 0: # Avoid 0-division
            image = image / torch.max(image)

        if image.ndim == 2: # [H, W]
            image = image.unsqueeze(0)  # [1, H, W]

        # Turn BIRADS categories into labels (categorical encoding)
        birads_to_label = {
            'BI-RADS 1': 1,  # Negative
            'BI-RADS 2': 2,  # Benign
            'BI-RADS 3': 3,  # Benign for further inspection
            'BI-RADS 4': 4,  # Suspicious
            'BI-RADS 5': 5,  # Highly suggestive of malignancy
        }
        label = birads_to_label.get(category, -1)

        # Image as a Tensor + BIRADS category
        return image, label