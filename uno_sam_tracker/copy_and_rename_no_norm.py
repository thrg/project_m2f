import os
from tqdm import tqdm


def copy_and_rename_files_in_subdir(input_folder):
    for root, dirs, files in tqdm(os.walk(input_folder)):
        for file in files:
            # Construct the full file path
            old_file_path = os.path.join(root, file)

            new_file_name = file.replace("_no_norm", "")
            new_file_path = os.path.join(input_folder, new_file_name)

            os.rename(old_file_path, new_file_path)
            print(f"Copied and renamed: {old_file_path} -> {new_file_path}")


input_folder = '../datasets/street_obstacle_sequences/ood_scores'

copy_and_rename_files_in_subdir(input_folder)