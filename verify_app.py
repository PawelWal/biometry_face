from src import FaceVer, count_metrics
import click
import os
from sklearn.metrics import classification_report
from math import ceil
import matplotlib.pyplot as plt
import json


@click.command()
@click.option("--train_dir", "-t", required=True)
@click.option("--test_dir", "-e", required=True)
@click.option("--test_dir_unknown", "-e", required=True)
@click.option("--model_name", "-m", default="ArcFace")
@click.option("--detector_backend", "-db", default="ssd")
@click.option("--backbone", "-b", default="deepface")
@click.option("--classifier", "-c", default="DistanceClassifier")
@click.option("--decision_th", "-d", default=0.6)
@click.option("--bs", default=48)
def main(
    train_dir,
    test_dir,
    test_dir_unknown,
    model_name,
    detector_backend,
    backbone,
    classifier,
    decision_th,
    bs,
):
    print("Name", test_dir.split("/")[-2])
    app = FaceVer(
        model_name,
        backbone,
        detector_backend,
        classifier,
        decision_th
    )
    app.train(train_dir)
    count_metrics(
        app,
        test_dir,
        test_dir_unknown,
        bs
    )


main()
