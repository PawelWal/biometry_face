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
    img_arr = np.array(img_list)
    resp_objs = DeepFace.represent(
        img_path=img_arr,
        model_name=model_name,
        enforce_detection=False
    )
    vectors = [resp_obj["embedding"] for resp_obj in resp_objs]
    return vectors
