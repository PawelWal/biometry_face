from deepface import DeepFace
import click
import os
from sklearn.metrics import classification_report
from time import time


DATADIR = "../subsamples"
MODELS = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace", #4
  "DeepID",
  "ArcFace", #6
  "Dlib",
  "SFace",
  "GhostFaceNet"
]
METRICS = ["cosine", "euclidean", "euclidean_l2"]

def evaluate(
    model_id,
    metric_id,
    ds_name
):
    train_dir = os.path.join(DATADIR, ds_name, "train")
    test_dir = os.path.join(DATADIR, ds_name, "test")

    train_files = [f.split("_")[0] for f in os.listdir(train_dir) if f.endswith(".jpg")]
    test_files = [f for f in os.listdir(test_dir)]
    new_files = set(test_files) - set(train_files)
    print(new_files)

    y_true = []
    y_pred = []
    start = time()
    for dir in os.listdir(test_dir):
        for file in os.listdir(f"{test_dir}/{dir}"):
            try:
                dfs = DeepFace.find(img_path=f"{test_dir}/{dir}/{file}",
                                db_path = train_dir,
                                distance_metric = METRICS[0],
                                model_name = MODELS[model_id],
                                silent=True,
                                enforce_detection=False,
                        )
                res = dfs[0].to_dict(orient="list")
                res = {k: v[0] if len(v) > 0 else None for k, v in res.items() if k in ["identity", "distance"]}['identity']
                pred = res.split('/')[-1].split('_')[0] if res else -1
            except Exception as e:
                print("Could not find a match for", f"{test_dir}/{dir}/{file}", e)
                pred = -1

            y_true.append(int(dir))
            if dir in new_files and pred == -1:
                pred = int(dir)
            y_pred.append(int(pred))
    print("Time taken", time() - start)
    print(y_true)
    print(y_pred)

    report = classification_report(y_true, y_pred)
    print(report)

    # for file in os.listdir(train_dir):
    #     pid = file.split("person")[1].split("_")[0]
    #     print(file, pid)
        # for model in MODELS:
        #     print(model)
        #     DeepFace.build_model(files, model_name=model, enable_face_analysis=True)
        #     print("Model built successfully")


@click.command()
@click.option(
    "--model_id",
    help="Model id",
    default=4
)
@click.option(
    "--metric_id",
    help="Metric id",
    default=0
)
@click.option(
    "--ds_name",
    help="Dataset name",
    default="facescrub"
)
def main(
    model_id,
    metric_id,
    ds_name
):
    evaluate(
        model_id,
        metric_id,
        ds_name
    )


main()
