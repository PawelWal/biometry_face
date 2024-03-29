from deepface import DeepFace
import numpy as np


def build_representation(
    img_list,
    model_name="ArcFace",
    method="deepface"
):
    if method == "deepface":
        return build_representation_deepface(img_list, model_name)
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def build_representation_deepface(
    img_list,
    model_name="ArcFace"
):
    resp_objs = []
    for img in img_list:
        resp_objs.append(DeepFace.represent(
            img_path=img,
            model_name=model_name,
            enforce_detection=False
        ))
    vectors = [resp_obj["embedding"] for resp_obj in resp_objs]
    return vectors
