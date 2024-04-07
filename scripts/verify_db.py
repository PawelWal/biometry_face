from deepface import DeepFace
import click
import os
from time import time

DATADIR = f"{os.getcwd()}/../samples"
MODELS = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace", #4
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]

METRICS = ["cosine", "euclidean", "euclidean_l2"]


def verify_db(
    train_dir,
    test_dir
):
    model_name = MODELS[-1]
    print(f"MODEL NAME {model_name}")
    for file in os.listdir(test_dir):
        print(f"File: {file}")
        # start = time()
        dfs = DeepFace.find(img_path=f"{test_dir}/{file}",
                db_path = train_dir,
                distance_metric = METRICS[0],
                threshold = 0.6,
                silent=True
        )
        # dfs = DeepFace.verify(img1_path=f"{test_dir}/{file}",
        #         db_path = train_dir,
        #         distance_metric = METRICS[0]
        # )
        # print(f"Time taken: {time()-start}")
        # print(dfs)
        res = dfs[0].to_dict(orient="list")
        res = {k: v[0] if len(v) > 0 else None for k, v in res.items() if k in ["identity", "distance"]}
        # print(dfs[0].to_dict(orient="list"))
        print(res)


verify_db(f"{DATADIR}/train", f"{DATADIR}/test")
