import os
import cv2
from tqdm import tqdm
import click


def file_generator(dir_path):
    for root, _, files in os.walk(dir_path):
        for name in files:
            root_path = os.path.join(root, name)
            yield root_path, os.path.relpath(root_path, dir_path)


def rescale(img, to_shape):
    return cv2.resize(img, (to_shape[1], to_shape[0]), interpolation = cv2.INTER_AREA)


def get_scale_from_img(img):
    return img.shape[0], img.shape[1]


def process(scale_path, data_dir, output_dir):
    scale_img = cv2.imread(scale_path)
    scale = get_scale_from_img(scale_img)
    print(f"Scale is {scale}")

    for data_file, rel_path in tqdm(file_generator(data_dir)):
        img = cv2.imread(data_file)
        before_scale = get_scale_from_img(img)
        print(f"Before scale is {before_scale}")
        if img is None:
            continue

        img = rescale(img, scale)
        rescaled = get_scale_from_img(img)
        print(f"Rescaled is {rescaled}")

        cv2.imwrite(os.path.join(output_dir, rel_path), img)


@click.command()
@click.option('--scale_path', default="./data/scale_3.jpg", required=True, help='Path to the scale image')
@click.option('--data_dir', default="./data_my", required=True, help='Path to the data directory')
@click.option('--output_dir', default="./data_my_rescale", required=True, help='Path to the output directory')
def main(scale_path, data_dir, output_dir):
    print(f"Running with scale_path={scale_path}, data_dir={data_dir}, output_dir={output_dir}")
    process(scale_path, data_dir, output_dir)

main()

        