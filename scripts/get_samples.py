import os
import shutil


def get_samples():
    base = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_subset"
    train_path = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_subset/train"
    test_path = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_subset/test1"
    test1_path = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_subset/test_unkown"

    # new_samples_path_train = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_new/train"
    # new_samples_path = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_new/test_known"
    # new_samples_path_unknown = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_new/test_unknown"
    # dest_path = "/home/bwalkow/repos/biometry_face/dataset/processed_celeb_test_cls"

    train_cls = []
    for cls in os.listdir(train_path):
        train_cls.append(int(cls))
    old_test = []
    for cls in os.listdir(test_path):
        for i, file in enumerate(os.listdir(os.path.join(test_path, cls))):
            # old_test.append(f"{cls}/{file}")
            if i < 3:
                os.makedirs(os.path.join(base, "dev_known", cls), exist_ok=True)
                shutil.copyfile(
                    os.path.join(test_path, cls, file),
                    os.path.join(base, "dev_known", cls, file)
                )
                os.remove(os.path.join(test_path, cls, file))
    old_unknown = []
    for cls in os.listdir(test1_path):
        for i, file in enumerate(os.listdir(os.path.join(test1_path, cls))):
            if i < 3:
                os.makedirs(os.path.join(base, "dev_unknown", cls), exist_ok=True)
                shutil.copyfile(
                    os.path.join(test1_path, cls, file),
                    os.path.join(base, "dev_unknown", cls, file)
                )
                os.remove(os.path.join(test1_path, cls, file))




    # new_cls = []
    # z = 0
    # for cls in os.listdir(new_samples_path):
    #     if int(cls) in train_cls and z < 100:
    #         z += 1
    #         for file in os.listdir(os.path.join(new_samples_path, cls)):
    #             if f"{cls}/{file}" not in old_test:
    #                 os.makedirs(os.path.join(dest_path, "test", cls), exist_ok=True)
    #                 shutil.copyfile(os.path.join(new_samples_path, cls, file), os.path.join(dest_path, "test", cls, file))

    # i, j = 0, 0
    # for cls in os.listdir(new_samples_path):
    #     if int(cls) in train_cls and i < 100:
    #         i += 1
    #         for file in os.listdir(os.path.join(new_samples_path, cls)):
    #             if f"{cls}/{file}" not in old_test:
    #                 os.makedirs(os.path.join(dest_path, "test", cls), exist_ok=True)
    #                 shutil.copyfile(os.path.join(new_samples_path, cls, file), os.path.join(dest_path, "test", cls, file))

    # for cls in os.listdir(new_samples_path_unknown):
    #     if int(cls) not in train_cls and j < 20:
    #         j += 1
    #         for file in os.listdir(os.path.join(new_samples_path_unknown, cls)):
    #             if f"{cls}/{file}" not in old_unknown:
    #                 os.makedirs(os.path.join(dest_path, "test_unknown", cls), exist_ok=True)
    #                 shutil.copyfile(os.path.join(new_samples_path_unknown, cls, file), os.path.join(dest_path, "test_unknown", cls, file))



get_samples()
