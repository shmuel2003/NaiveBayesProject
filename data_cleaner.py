from typing import List, Dict

class DataCleaner:
    def clean(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned_data = []
        for row in data:
            cleaned_row = {}
            valid = True
            for key, value in row.items():
                if value is None or value.strip() == "":
                    value = "unknown"
                value = value.strip().lower()
                cleaned_row[key] = value
            if valid:
                cleaned_data.append(cleaned_row)
        return cleaned_data