from dataclasses import dataclass

@dataclass
class amharic_dataset:
    dataset: str = "amharic_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/data/fine_tun_data2.json"