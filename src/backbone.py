from deepface import DeepFace
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import torch
from src import build_model
from face_alignment import align


adaface_models = {
    'ir_18':"/mnt/data/pwalkow/biometria/models/adaface_ir18_webface4m.ckpt",
}

def build_representation(
    img_list,
    model_name="ArcFace",
    method="deepface",
    detector_backend="opencv",
    verbose=False
):
    if method == "deepface":
        return build_representation_deepface(img_list, model_name, detector_backend, verbose)
    elif method == "adaface":
        return build_representation_adaface(img_list, model_name, verbose)
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


def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor


def build_representation_adaface(
    img_list,
    model,
    verbose=False
):
    model = load_pretrained_model('ir_18')

    features = []
    for img in tqdm(img_list):
        aligned_rgb_img = align.get_aligned_face(img)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    return features
