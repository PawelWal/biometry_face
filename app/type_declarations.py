"""Declarations of used types."""
from pydantic import BaseModel

class BiomUser(BaseModel):
    """User in system."""

    user_dir: str


class BiomVerify(BaseModel):
    """User to verify."""

    user_img : str
    user_cls : str


class BiomIdentify(BaseModel):
    """User to identify."""

    user_img : str
