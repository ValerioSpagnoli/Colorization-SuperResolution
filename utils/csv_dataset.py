import os
import csv

# Define dataset root directory
dataset_root = "./data/"

# Define original, reference, and restored folders
img_folder = os.path.join(dataset_root, "img")
ref_folder = os.path.join(dataset_root, "ref")
res_folder = os.path.join(dataset_root, "res")

# Create a list of files of the respective folders
img_files = sorted(os.listdir(img_folder), key=lambda x: int(os.path.splitext(x)[0]))
ref_files = sorted(os.listdir(ref_folder), key=lambda x: int(os.path.splitext(x)[0]))
res_files = sorted(os.listdir(res_folder), key=lambda x: int(os.path.splitext(x)[0]))

# Create and write csv file
csv_file = open(dataset_root + "dataset.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["img_path", "ref_path", "res_path"])

# Iterate over files and write their paths in the i-th row
for i in range(len(img_files)):
    csv_writer.writerow([
        os.path.join("img", img_files[i]), 
        os.path.join("ref", ref_files[i]), 
        os.path.join("res", res_files[i])])

# Close file
csv_file.close()
