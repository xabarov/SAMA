import os


def create_yaml(yaml_short_name, save_folder, label_names, dataset_name='Dataset', use_test=None):
    yaml_full_name = os.path.join(save_folder, yaml_short_name)
    with open(yaml_full_name, 'w') as f:
        f.write(f"# {dataset_name}\n")
        # Paths:
        path_str = f"path: {save_folder}\n"
        path_str += "train: images/train  # train images (relative to 'path') \n"
        path_str += "val: images/val  # val images (relative to 'path')\n"
        if not use_test:
            path_str += "test:  # test images (optional)\n"
        else:
            path_str += "test:  images/test # test images\n"
        f.write(path_str)
        # Classes:
        f.write("#Classes\n")
        f.write(f"nc: {len(label_names)} # number of classes\n")
        f.write(f"names: {label_names}\n")
