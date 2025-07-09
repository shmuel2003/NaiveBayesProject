import csv

class DataLoader:
    def load_csv(self, filepath):
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            return list(reader)