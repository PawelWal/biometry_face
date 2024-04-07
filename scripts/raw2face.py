import os
import cv2
from tqdm import tqdm
import click
import dlib


def file_generator(dir_path):
    for root, _, files in os.walk(dir_path):
        for name in files:
            root_path = os.path.join(root, name)
            yield root_path, os.path.relpath(root_path, dir_path)


def get_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
    return x, y, w, h


def cut_face(image_path, output_path):
    x, y, w, h = get_face(image_path)
    image = cv2.imread(image_path)
    # Cut the frame plus some margin
    margin_w = 150
    margin_h = 200
    face = image[y-margin_h:y+h+margin_h, x-margin_w:x+w+margin_w]
    cv2.imwrite(output_path, face)


def cut_center(image_path, output_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    x, y = w//2, h//2
    margin = 1300
    img = img[y-margin:y+margin, x-margin:x+margin]
    
    cv2.imwrite(output_path, img)

def process(data_dir, output_dir):
    for data_file, rel_path in tqdm(file_generator(data_dir)):
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cut_face(data_file, output_path)


@click.command()
@click.option('--data_dir', default="./data_raw", required=True, help='Path to the data directory')
@click.option('--output_dir', default="./data_my_face", required=True, help='Path to the output directory')
def main(data_dir, output_dir):
    print(f"Running with data_dir={data_dir}, output_dir={output_dir}")
    process(data_dir, output_dir)


main()
# cut_center('./data_raw/BW_5.jpg', './data_raw/BW_5_fixed.jpg')

