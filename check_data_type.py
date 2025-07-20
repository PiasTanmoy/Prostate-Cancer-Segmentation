import os
import pandas as pd

img_root = "input/images"  # Change this to your images root directory

file_types = ["adc", "cor", "hbv", "sag", "t2w"]
records = []

for patient_id in sorted(os.listdir(img_root)):
    print(f"Processing patient: {patient_id}")
    patient_folder = os.path.join(img_root, patient_id)
    if not os.path.isdir(patient_folder):
        continue
    # Collect all files in this patient folder
    files = os.listdir(patient_folder)
    # For each file, parse study_id and file_type
    study_files = {}
    for f in files:
        if not f.endswith('.mha'):
            continue
        parts = f.split('_')
        if len(parts) < 3:
            continue
        pid, study_id, file_type_ext = parts[0], parts[1], parts[2]
        file_type = file_type_ext.replace('.mha', '')
        if study_id not in study_files:
            study_files[study_id] = {ft: 0 for ft in file_types}
        if file_type in file_types:
            study_files[study_id][file_type] = 1
    # Save a row for each study_id
    for study_id, type_dict in study_files.items():
        total_format = sum(type_dict.values())
        row = {
            "patient_id": patient_id,
            "study_id": study_id,
            **type_dict,
            "total_format": total_format
        }
        records.append(row)
        print(row)

df = pd.DataFrame(records)
df = df[["patient_id", "study_id"] + file_types + ["total_format"]]
df.to_csv("data_file_type.csv", index=False)
print("Saved to image_file_summary.csv")
