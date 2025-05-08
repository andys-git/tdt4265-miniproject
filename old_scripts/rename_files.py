import os

def rename_files_in_directory(directory, search="_overlay", replace=""):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if search in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(search, replace)
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# Example usage:
if __name__ == "__main__":
    directory_path = r"/final_result_overlay_full"
    rename_files_in_directory(directory_path)
