import os
import re

def rename_files_in_folder(folder_path):
    # Dictionary to count occurrences of new filenames
    filename_counts = {}
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            match = re.search(r'(\d{5})\.png$', filename)
            if match:
                base_name = match.group(1)
                # Count occurrences to handle duplicates
                count = filename_counts.get(base_name, 0)
                filename_counts[base_name] = count + 1
                if count > 0:
                    new_filename = f"{base_name}_{count}.png"
                else:
                    new_filename = f"{base_name}.png"
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, new_filename)
                os.rename(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'")
            else:
                print(f"Filename '{filename}' does not match the expected pattern.")
        else:
            print(f"Skipping '{filename}': not a .png file.")

# Example usage:
folder_path = '/Users/sigvard/Downloads/compiled_result/ffhq/non-linear/scale_05/final'  # Replace with the path to your folder
rename_files_in_folder(folder_path)
