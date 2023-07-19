import os
import pandas as pd


class CatsAndDogsCSV:
    def __init__(self, csv_file, root_dir):
        self.csv_file = csv_file
        self.root_dir = root_dir

        if not os.path.exists(self.csv_file):
            self.generate_csv()

    def generate_csv(self):
        files = os.listdir(self.root_dir)
        annotations = []

        for file_name in files:
            if file_name.endswith('.jpg'):
                label = 0 if file_name.startswith('cat') else 1
                annotations.append({'filename': file_name, 'label': label})

        annotations_df = pd.DataFrame(annotations)
        annotations_df.to_csv(self.csv_file, index=False)
        print(f"CSV file '{self.csv_file}' created successfully.")
