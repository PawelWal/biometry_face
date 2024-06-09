import os
import shutil
from collections import defaultdict

DATASETS_DIR = "/mnt/data/pwalkow/biometria"


def join_datasets(
    to_join=[
        "processed_celeb_subset_v2",
        "processed_celeb_subset_v2_mod",
        "processed_celeb_subset_v2_mod_snap",
        # "tinyface_splitted1"
    ],
    dest="processed_celeb_subset_v2_all",
    # tiny_face_cls=0,
):
    for dir in ["train", "test_known", "dev_known", "test_unknown", "dev_unknown"]:
        os.makedirs(os.path.join(DATASETS_DIR, dest, dir), exist_ok=True)

    split_sizes = defaultdict(int)

    for split in ["train", "test_known", "dev_known", "test_unknown", "dev_unknown"]:
        for dataset in to_join:
            for cls in os.listdir(os.path.join(DATASETS_DIR, dataset, split)):
                os.makedirs(os.path.join(DATASETS_DIR, dest, split, cls), exist_ok=True)
                for file in os.listdir(os.path.join(DATASETS_DIR, dataset, split, cls)):
                    shutil.copyfile(
                        os.path.join(DATASETS_DIR, dataset, split, cls, file),
                        os.path.join(DATASETS_DIR, dest, split, cls, file)
                    )
                    if "mod" in dataset:
                        split_sizes[f"{split}_mod"] += 1
                    else:
                        split_sizes[f"{split}_norm"] += 1

    print(split_sizes)


join_datasets()
