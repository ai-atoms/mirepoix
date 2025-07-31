import os

# Define the mapping dictionary
equiv_dic_fpa70 = {
    f'{4000*k}_8_ry.png': f'72_000_{str(1000000 + k + 1)[1:]}.png' 
    for k in range(152)
}

# Target folder
folder_path = 'data/datasets/d567/images_temp'

# Rename files based on the dictionary
for old_name, new_name in equiv_dic_fpa70.items():
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        print(f"File not found: {old_name}")
