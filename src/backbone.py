from deepface import DeepFace
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def build_representation(
    img_list,
    model_name="ArcFace",
    method="deepface",
    detector_backend="opencv",
    verbose=False
):
    if method == "deepface":
        return build_representation_deepface(img_list, model_name, detector_backend, verbose)
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def build_representation_deepface(
    img_list,
    model_name="ArcFace",
    detector_backend="opencv",
    verbose=False
):
    resp_objs = []
    # with tf.device('/device:GPU:1'):
    if verbose:
        for img in tqdm(img_list):
            vector = DeepFace.represent(
                img_path=img,
                model_name=model_name,
                enforce_detection=False,
                detector_backend=detector_backend
            )
            if len(vector) == 0:
                # resp_objs.append({"embedding": np.zeros(512)}) # dummy fix
                resp_objs.append({"embedding": None})
            elif len(vector) < 512:
                resp_objs.append(vector[0])
            else:
                resp_objs.append(vector)
    else:
        for img in img_list:
            vector = DeepFace.represent(
                img_path=img,
                model_name=model_name,
                enforce_detection=False,
                detector_backend=detector_backend
            )
            if len(vector) == 0:
                resp_objs.append({"embedding": np.zeros(512)}) # dummy fix
                # resp_objs.append({"embedding": None})
            elif len(vector) < 512:
                resp_objs.append(vector[0])
            else:
                resp_objs.append(vector)
    vectors = [resp_obj["embedding"] for resp_obj in resp_objs]
    return vectors
