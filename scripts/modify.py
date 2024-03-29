"""Autor: Bartosz Walkowiak"""
import os
import cv2
from tqdm import tqdm
from enum import Enum
import click
import numpy as np
from copy import deepcopy

class TransformationType(Enum):
    BLUR = "blur"
    LUM = "lum"


class Threshold:
    def __init__(self, from_value, to_value):
        self.from_value = from_value
        self.to_value = to_value

    def is_satisfied(self, value):
        return self.from_value <= value <= self.to_value
    
    def __str__(self) -> str:
        return f"{self.from_value}-{self.to_value}"
    
    def __repr__(self) -> str:
        return f"{self.from_value}-{self.to_value}"
    

class Scale:
    def __init__(self, factor=0):
        self.factor = factor

    def scale(self, value):
        pass

class QuadraticScale(Scale):
    def __init__(self, factor=0):
        super().__init__(factor)

    def scale(self, value):
        return value ** 2
    
    def __str__(self):
        return "QuadraticScale"
    
    def __repr__(self):
        return "QuadraticScale"
    

class LinearScale(Scale):
    def __init__(self, factor=0):
        super().__init__(factor)

    def scale(self, value):
        return value * self.factor
    
    def __str__(self):
        return f"LinearScale_({round(self.factor, 2)})"
    
    def __repr__(self):
        return f"LinearScale_({round(self.factor, 2)})"
    
class ConstantScale(Scale):
    def __init__(self, factor=0):
        super().__init__(factor)

    def scale(self, value):
        return value + self.factor
    
    def __str__(self):
        return f"ConstantScale_({round(self.factor, 2)})"
    
    def __repr__(self):
        return f"ConstantScale_({round(self.factor, 2)})"
        
    

PSNR_NOISE_MAP = {
    50: 0.1,
    40: 0.8,
    30: 4.0,
    20: 10.0,
    10: 50.0
}

TEST_FOR = {
    TransformationType.BLUR: [
        Threshold(50, 80),
        Threshold(40, 50),
        Threshold(30, 40),
        Threshold(20, 30),
        Threshold(10, 20)
    ],
    TransformationType.LUM: [
        QuadraticScale(),
        LinearScale(factor=0.5),
        LinearScale(factor=0.6),
        LinearScale(factor=0.75),
        LinearScale(factor=4/3),
        LinearScale(factor=1.5),
        ConstantScale(factor=-100),
        ConstantScale(factor=-20),
        ConstantScale(factor=-10),
        ConstantScale(factor=30)
    ]
}


class Transformation:
    def transform(self, img):
        pass

class BlurTransformation(Transformation):
    def transform(self, img, psnr_threshold=Threshold(20, 30)):
        noise_level = PSNR_NOISE_MAP[psnr_threshold.from_value]
        noise = np.random.poisson(noise_level, img.shape).astype(np.uint8)
        result_img = cv2.add(deepcopy(img), noise)
        
        if psnr_threshold.is_satisfied(cv2.PSNR(img, result_img)):
            return result_img
        else:
            print(f"{cv2.PSNR(img, result_img)} for {psnr_threshold}, noise_level {noise_level}")
            raise Exception("PSNR is not satisfied")
        

class LumTransformation(Transformation):
    def transform(self, img, scale=LinearScale(factor=0.5)):
        result_img = deepcopy(img)
        img_yuv = cv2.cvtColor(result_img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:,:,0]
        y_channel = np.array(y_channel, dtype=np.float32)
        
        y_channel = scale.scale(y_channel)

        if isinstance(scale, QuadraticScale):
            # min max scaling
            min_val = np.min(y_channel)
            max_val = np.max(y_channel)
            y_channel = (y_channel - min_val) / (max_val - min_val) * 255
        else:
            y_channel = np.clip(y_channel, 0, 255)
        
        img_yuv[:,:,0] = np.array(y_channel, dtype=np.uint8)

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)



def file_generator(dir_path):
    for root, _, files in os.walk(dir_path):
        for name in files:
            root_path = os.path.join(root, name)
            yield root_path, os.path.relpath(root_path, dir_path)


def modify(data_dir, transfomation_type, output_dir):
    for data_file, rel_path in tqdm(file_generator(data_dir)):
        for th in TEST_FOR[transfomation_type]:
            try:
                img = cv2.imread(data_file)
                if img is None:
                    continue
                if transfomation_type == TransformationType.BLUR:
                    img = BlurTransformation().transform(img, th)
                elif transfomation_type == TransformationType.LUM:
                    img = LumTransformation().transform(img, th)
                else:
                    print("Unknown transformation")
            except Exception as e:
                print("Exception", e)
                continue
            output_file = os.path.join(output_dir, str(th), rel_path)
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            cv2.imwrite(output_file, img)



@click.command()
@click.option('--data_dir', default="./data",required=True, help='Path to the data directory')
@click.option('--tr', default="blur", required=True, help='Transformation type', type=click.Choice([t.value for t in TransformationType]))
@click.option('--output_dir', default="./data_mod", required=True, help='Path to the output directory')
def main(data_dir, tr, output_dir):
    print(f"Running with data_dir={data_dir}, transformation={tr}, output_dir={output_dir}")
    modify(data_dir, TransformationType(tr), output_dir)


main()
    