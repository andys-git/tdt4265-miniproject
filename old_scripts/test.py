import os

def check_jpg_txt_pairs(directory):
    # List all files in the given directory
    files = os.listdir(directory)

    # Separate files into jpg and txt sets (without extension)
    jpg_files = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.jpg')}
    txt_files = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.txt')}

    # Check for .jpg files missing a .txt
    missing_txt = jpg_files - txt_files
    if missing_txt:
        print("Missing .txt files for the following .jpg files:")
        for name in missing_txt:
            print(f"  {name}.jpg")
    else:
        print("All .jpg files have corresponding .txt files.")

    # Check for .txt files missing a .jpg
    missing_jpg = txt_files - jpg_files
    if missing_jpg:
        print("Missing .jpg files for the following .txt files:")
        for name in missing_jpg:
            print(f"  {name}.txt")
    else:
        print("All .txt files have corresponding .jpg files.")

# Replace with your directory path
check_jpg_txt_pairs("C:\\Users\\andys\Documents\TDT4265\clean_data\\aalesund")
check_jpg_txt_pairs("C:\\Users\\andys\Documents\TDT4265\clean_data\\aftergoal")
check_jpg_txt_pairs("C:\\Users\\andys\Documents\TDT4265\clean_data\\hamkam")
