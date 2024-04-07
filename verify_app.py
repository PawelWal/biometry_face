from src import FaceVer, count_metrics
import click
import os
from sklearn.metrics import classification_report
from math import ceil
import matplotlib.pyplot as plt
import json


@click.command()
@click.option("--data_dir", "-t", required=True)
@click.option("--model_name", "-m", default="ArcFace")
@click.option("--detector_backend", "-db", default="ssd")
@click.option("--backbone", "-b", default="deepface")
@click.option("--classifier", "-c", default="DistanceClassifier")
@click.option("--decision_th", "-d", default=0.5)
@click.option("--bs", default=48)
@click.option("--exclude_unknown", "-eu", is_flag=True)
def main(
    data_dir,
    model_name,
    detector_backend,
    backbone,
    classifier,
    decision_th,
    bs,
    exclude_unknown=False
):
    print("Name", test_dir.split("/")[-2])
    app = FaceVer(
        model_name,
        backbone,
        detector_backend,
        classifier,
        decision_th
    )
    train_dir = os.path.join(data_dir, "train")
    app.train(train_dir)
    test_dir = os.path.join(data_dir, "test_known")
    test_dir_unknown = os.path.join(data_dir, "test_unkown")
    dev_dir = os.path.join(data_dir, "dev_known")
    dev_dir_unknown = os.path.join(data_dir, "dev_unknown")
    count_metrics(
        app,
        test_dir,
        test_dir_unknown if not exclude_unknown else None,
        dev_dir,
        dev_dir_unknown if not exclude_unknown else None,
        bs
    )


main()
