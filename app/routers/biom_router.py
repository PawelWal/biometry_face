"""Rest "/biom" router."""
from fastapi import APIRouter, Body, HTTPException

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


@router.post("/add_user/",
             tags=["tasks"],
             summary="Add new user to the system")
def add_user(
        user: BiomUser = Body(
            example={
                "user_dir": "/home/test-user/imgs"
            }),
):
    """Adds new user to the system."""
    if not facever.is_training:
        return facever.add_user(user.user_dir)
    else:
        return WAITING_MSG


@router.post("/verify/",
                tags=["tasks"],
                summary="Verify user")
def verify_user(
        user: BiomVerify = Body(
            example={
                "user_img": "/home/test-user/imgs/test.jpg",
                "user_cls": "test-user"
            }),
):
    """Verifies user."""
    if not facever.is_training:
        return facever.verify(user.user_img, user.user_cls)
    else:
        return WAITING_MSG


@router.post("/identify/",
                tags=["tasks"],
                summary="Identify user")
def identify_user(
        user: BiomIdentify = Body(
            example={
                "user_img": "/home/test-user/imgs/test.jpg"
            }),
):
    """Identifies user."""
    if not facever.is_training:
        return facever.identify(user.user_img)
    else:
        return WAITING_MSG
