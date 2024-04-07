"""Rest "/biom" router."""
from fastapi import APIRouter, Body, HTTPException
import os

from ..type_declarations import BiomUser, BiomVerify, BiomIdentify
from ..router_utils import facever


router = APIRouter(
    prefix="/biom"
)

WAITING_MSG = "System is training, please wait."


@router.get("/")
async def root_tasks():
    """Endpoint for checking if task is alive."""
    return {"message": "Tasks alive"}


def check_img_path(img_path):
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return None

def unpack_ident(resp):
    classes, probs = resp[0], resp[1]
    return [{"class": cls, "prob": prob} for cls, prob in zip(classes, probs)]
    

@router.post("/add_user/",
             tags=["tasks"],
             summary="Add new user to the system")
def add_user(
        user: BiomUser = Body(
            example={
                "user_dir": "./dataset/processed_celeb_new/train/1002"
            }),
):
    """Adds new user to the system."""
    check_img_path(user.user_dir)
    if not facever.is_training:
        resp = facever.add_user(user.user_dir)
        return {"new_class": resp}
    else:
        return WAITING_MSG


@router.post("/verify/",
                tags=["tasks"],
                summary="Verify user")
def verify_user(
        user: BiomVerify = Body(
            example={
                "user_img": "./dataset/processed_celeb_subset/test/200/BW_8.jpg",
                "user_cls": 200
            }),
):
    """Verifies user."""
    check_img_path(user.user_img)
    if not facever.is_training:
        resp = facever.verify(user.user_img, user.user_cls)
        if isinstance(resp, list):
            return resp[0]
        return resp
    else:
        return WAITING_MSG

@router.post("/identify/",
                tags=["tasks"],
                summary="Identify user")
def identify_user(
        user: BiomIdentify = Body(
            example={
                "user_img": "./dataset/processed_celeb_subset/test/200/BW_8.jpg"
            }),
):
    """Identifies user."""
    check_img_path(user.user_img)
    if not facever.is_training:
        resp = facever.identify(user.user_img)
        if isinstance(resp, list):
            return unpack_ident(resp[0])
        return unpack_ident(resp)
    else:
        return WAITING_MSG


@router.post("/get_imgs/",
                tags=["tasks"],
                summary="Get images")
def get_imgs(
        user: BiomUser = Body(
            example={
                "user_dir": "./dataset/processed_celeb_subset/test/200"
            }),
):
    """Get images."""
    check_img_path(user.user_dir)
    return os.listdir(user.user_dir)


@router.get("/get_classes/",    
                tags=["tasks"],
                summary="Get classes")
def get_classes():
    """Get classes."""
    return sorted(list(set(facever.classes)))


@router.get("/get_num_classes/",    
                tags=["tasks"],
                summary="Get classes")
def get_num_classes():
    """Get classes."""
    return len(set(facever.classes))

