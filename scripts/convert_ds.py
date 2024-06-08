import os
import shutil


TINY_FACE_DIR = "/mnt/data/pwalkow/biometria/tinyface/Training_Set"
DEST_DIR = "/mnt/data/pwalkow/biometria/tinyface_splitted1"

def convert_tiny_face(
    min_files_per_cls=5,
    train_samples=0.4,
    test_samples=0.4,
    dev_samples=0.2,
):
    os.makedirs(DEST_DIR, exist_ok=True)
    train_dir = f"{DEST_DIR}/train"
    test_dir = f"{DEST_DIR}/test_known"
    dev_dir = f"{DEST_DIR}/dev_known"
    test_unk_dir = f"{DEST_DIR}/test_unknown"
    dev_unk_dir = f"{DEST_DIR}/dev_unknown"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_unk_dir, exist_ok=True)
    os.makedirs(dev_unk_dir, exist_ok=True)

    cls = []
    cls_unknown = []
    for d in os.listdir(TINY_FACE_DIR):
        if len(os.listdir(f"{TINY_FACE_DIR}/{d}")) >= min_files_per_cls:
            cls.append(d)
        else:
            cls_unknown.append(d)

    # train_cls = set([dir for dir in os.listdir(TINY_FACE_DIR)])
    # test_cls = set([dir for dir in os.listdir(TINY_FACE_DIR)])
    # dev_cls = set([dir for dir in os.listdir(TINY_FACE_DIR)])

    print(f"Cls: {len(cls)}, unknown: {len(cls_unknown)}")

    test_files = 0
    for class_dir in cls:
        os.makedirs(f"{train_dir}/{class_dir}", exist_ok=True)
        os.makedirs(f"{test_dir}/{class_dir}", exist_ok=True)
        file_nb = len(os.listdir(f"{TINY_FACE_DIR}/{class_dir}"))
        for j, file in enumerate(os.listdir(f"{TINY_FACE_DIR}/{class_dir}")):
            if j < train_samples * file_nb:
                shutil.copy(f"{TINY_FACE_DIR}/{class_dir}/{file}", f"{train_dir}/{class_dir}/{file}")
            elif j < train_samples * file_nb + test_samples * file_nb:
                shutil.copy(f"{TINY_FACE_DIR}/{class_dir}/{file}", f"{test_dir}/{class_dir}/{file}")
                test_files += 1
            else:
                os.makedirs(f"{dev_dir}/{class_dir}", exist_ok=True)
                shutil.copy(f"{TINY_FACE_DIR}/{class_dir}/{file}", f"{dev_dir}/{class_dir}/{file}")

    test_unk_files = 0
    for class_dir in cls_unknown:
        os.makedirs(f"{test_unk_dir}/{class_dir}", exist_ok=True)
        os.makedirs(f"{dev_unk_dir}/{class_dir}", exist_ok=True)
        file_nb = len(os.listdir(f"{TINY_FACE_DIR}/{class_dir}"))
        for j, file in enumerate(os.listdir(f"{TINY_FACE_DIR}/{class_dir}")):
            if j < test_samples * file_nb:
                shutil.copy(f"{TINY_FACE_DIR}/{class_dir}/{file}", f"{test_unk_dir}/{class_dir}/{file}")
                test_unk_files += 1
            else:
                shutil.copy(f"{TINY_FACE_DIR}/{class_dir}/{file}", f"{dev_unk_dir}/{class_dir}/{file}")

    print("Done")
    print(f"Test files: {test_files}")
    print(f"Test unknown files: {test_unk_files}")


convert_tiny_face()
