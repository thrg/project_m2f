import os
from tqdm import tqdm


delete_number = "000300"
input_folder = '../datasets/street_obstacle_sequences/raw_data_tmp'

if not os.path.exists(input_folder):
    raise FileNotFoundError(f"The directory {input_folder} does not exist.")

for root, dirs, files in tqdm(os.walk(input_folder)):
    for file in files:
        if file.split(".")[0] > delete_number:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Removed: {file_path}")