import os
import shutil

DATASET_DIR = "/mnt/data/pwalkow/biometria/processed_celeb_new"
DEST_DIR = "/mnt/data/pwalkow/biometria/processed_celeb_subset_v2"


def get_subset(known_cls=130, unknown_cls=40, dev_sample=0.5):
    os.makedirs(DEST_DIR, exist_ok=True)

    known = os.listdir(os.path.join(DATASET_DIR, "train"))[:known_cls]
    unknown = os.listdir(os.path.join(DATASET_DIR, "test_unknown"))[:unknown_cls]

    train_samples = 0
    test_known_samples, dev_known_samples = 0, 0
    test_unknown_samples, dev_unknown_samples = 0, 0

    for cls in known:
        os.makedirs(os.path.join(DEST_DIR, "train", cls), exist_ok=True)
        for file in os.listdir(os.path.join(DATASET_DIR, "train", cls)):
            shutil.copyfile(
                os.path.join(DATASET_DIR, "train", cls, file),
                os.path.join(DEST_DIR, "train", cls, file)
            )
            train_samples += 1
        os.makedirs(os.path.join(DEST_DIR, "test_known", cls), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "dev_known", cls), exist_ok=True)
        for i, file in enumerate(os.listdir(os.path.join(DATASET_DIR, "test_known", cls))):
            if i < dev_sample * len(os.listdir(os.path.join(DATASET_DIR, "test_known", cls))):
                shutil.copyfile(
                    os.path.join(DATASET_DIR, "test_known", cls, file),
                    os.path.join(DEST_DIR, "test_known", cls, file)
                )
                test_known_samples += 1
            else:
                shutil.copyfile(
                    os.path.join(DATASET_DIR, "test_known", cls, file),
                    os.path.join(DEST_DIR, "dev_known", cls, file)
                )
                dev_known_samples += 1

    for cls in unknown:
        os.makedirs(os.path.join(DEST_DIR, "test_unknown", cls), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "dev_unknown", cls), exist_ok=True)
        for i, file in enumerate(os.listdir(os.path.join(DATASET_DIR, "test_unknown", cls))):
            if i < dev_sample * len(os.listdir(os.path.join(DATASET_DIR, "test_unknown", cls))):
                shutil.copyfile(
                    os.path.join(DATASET_DIR, "test_unknown", cls, file),
                    os.path.join(DEST_DIR, "test_unknown", cls, file)
                )
                test_unknown_samples += 1
            else:
                shutil.copyfile(
                    os.path.join(DATASET_DIR, "test_unknown", cls, file),
                    os.path.join(DEST_DIR, "dev_unknown", cls, file)
                )
                dev_unknown_samples += 1

    print(f"Train samples: {train_samples}")
    print(f"Test known samples: {test_known_samples}")
    print(f"Dev known samples: {dev_known_samples}")
    print(f"Test unknown samples: {test_unknown_samples}")
    print(f"Dev unknown samples: {dev_unknown_samples}")


get_subset()
