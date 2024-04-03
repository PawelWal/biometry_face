from src import FaceVer

def start_facever(
    model_name="ArcFace",
    detector_backend="ssd",
    backbone = "deepface",
    classifier = "DistanceClassifier",
    decision_th=0.95,
):
    """Starts face verification system."""
    app = FaceVer(
            model_name,
            backbone,
            detector_backend,
            classifier,
            decision_th
        )
    return app


facever = start_facever()
