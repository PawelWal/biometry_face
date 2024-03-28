import os
import json
import shutil
import click


BASE_DIR = ".."


def prepare_datasets(
    train_samples=20,
    test_samples=10,
    test_new_users=1,
    ds_name="facescrub",
    max_samples_train=1,
    max_samples_test=5,
    flat_train=True
):
    dataset_dir=f"{BASE_DIR}/{ds_name}"
    train_dir = f"{dataset_dir}/train"
    test_dir = f"{dataset_dir}/validate"
    print(f"Train dir: {train_dir}, test dir: {test_dir}")
    dest_train = f"{BASE_DIR}/subsamples/{ds_name}/train"
    dest_test = f"{BASE_DIR}/subsamples/{ds_name}/test"
    os.makedirs(dest_train, exist_ok=True)
    os.makedirs(dest_test, exist_ok=True)

    train_cls = set([dir for dir in os.listdir(train_dir)])
    test_cls = set([dir for dir in os.listdir(test_dir)])
    common = train_cls.intersection(test_cls)
    to_train = list(common)[:train_samples]
    to_test = to_train[:test_samples-test_new_users]
    to_test_new = list(test_cls-set(to_train))[:test_new_users]
    train_cls_mapping = {}
    print(f"Train: {to_train}, test: {to_test}, test_new: {to_test_new}")
    i = 0
    for dir in os.listdir(train_dir):
        if str(dir) in to_train:
            if flat_train:
                # os.makedirs(f"{dest_train}/{i}", exist_ok=True)
                train_cls_mapping[dir] = i
                for j, file in enumerate(os.listdir(f"{train_dir}/{dir}")):
                    if j < max_samples_train:
                        # print(f"{train_dir}/{dir}/{file} to {dest_train}/{i}/{j}.jpg")
                        shutil.copy(f"{train_dir}/{dir}/{file}", f"{dest_train}/{i}_{j}.jpg")
                    else:
                        break
                i += 1
            else:
                os.makedirs(f"{dest_train}/{i}", exist_ok=True)
                train_cls_mapping[dir] = i
                for j, file in enumerate(os.listdir(f"{train_dir}/{dir}")):
                    if j < max_samples_train:
                        # print(f"{train_dir}/{dir}/{file} to {dest_train}/{i}/{j}.jpg")
                        shutil.copy(f"{train_dir}/{dir}/{file}", f"{dest_train}/{i}/{j}.jpg")
                    else:
                        break
                i += 1

    train_max_idx = max(list(train_cls_mapping.values())) + 1

    test_used, test_new = 0, 0
    used = False
    i = 0
    for dir in os.listdir(test_dir):
        if dir in to_test:
            os.makedirs(f"{dest_test}/{train_cls_mapping[dir]}", exist_ok=True)
            for j, file in enumerate(os.listdir(f"{test_dir}/{dir}")):
                if j < max_samples_test:
                    # print(f"{test_dir}/{dir}/{file} to {dest_test}/{train_cls_mapping[dir]}/{j}.jpg")
                    shutil.copy(f"{test_dir}/{dir}/{file}", f"{dest_test}/{train_cls_mapping[dir]}/{j}.jpg")

        elif dir in to_test_new:
            os.makedirs(f"{dest_test}/{train_max_idx}", exist_ok=True)
            train_cls_mapping[dir] = train_max_idx
            for j, file in enumerate(os.listdir(f"{test_dir}/{dir}")):
                if j < max_samples_test:
                    # print(f"{test_dir}/{dir}/{file} to {dest_test}/{train_max_idx}/{j}.jpg")
                    shutil.copy(f"{test_dir}/{dir}/{file}", f"{dest_test}/{train_max_idx}/{j}.jpg")
            used = True
        if used:
            train_max_idx += 1
            used = False
        # print(f"Copying {file} to {dest_train}")
        # shutil.copyfile(f"{train_dir}/{file}", f"{dest_train}/{file}")
    with open(f"{BASE_DIR}/subsamples/{ds_name}/cls_mapping.json", "w") as f:
        f.write(json.dumps(train_cls_mapping, ensure_ascii=False))



@click.command()
@click.option("--train_samples", default=20, help="Number of training samples per class")
@click.option("--test_samples", default=10, help="Number of testing samples per class")
@click.option("--test_new_users", default=1, help="Number of new users in testing set")
@click.option("--ds_name", default="facescrub", help="Dataset name")
@click.option("--max_samples_train", default=1, help="Max number of samples per class in training set")
@click.option("--max_samples_test", default=5, help="Max number of samples per class in testing set")
@click.option("--flat_train", default=True, help="Whether to flatten the training set")
def main(
    train_samples,
    test_samples,
    test_new_users,
    ds_name,
    max_samples_train,
    max_samples_test,
    flat_train
):
    prepare_datasets(
        train_samples=train_samples,
        test_samples=test_samples,
        test_new_users=test_new_users,
        ds_name=ds_name,
        max_samples_train=max_samples_train,
        max_samples_test=max_samples_test,
        flat_train=flat_train
    )


main()
