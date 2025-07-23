import csv
import random
from typing import List, Dict

class DataLoader:
    def __init__(self, filename: str):
        self.filename = filename

    def load_data(self) -> List[Dict[str, str]]:
        with open(self.filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]

    def split_data(self, data: List[Dict[str, str]], train_ratio: float = 0.7):
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]