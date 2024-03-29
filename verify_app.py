from src import FaceVer
import click
import os
from sklearn.metrics import classification_report
from math import ceil


def count_metrics(
    train_dir,
    test_dir,
    model_name="ArcFace",
    backbone = "deepface",
    classifier = "SVMClassifier",
    decision_th=0.5,
    batch_size=24,
):
    app = FaceVer(
        model_name,
        backbone,
        classifier,
        decision_th
    )
    app.train(train_dir)

    app_classes = app.classifier.classes
    X_test, y_test = [], []
    for cls in os.listdir(test_dir):
        for img in os.listdir(os.path.join(test_dir, cls)):
            if cls in app_classes:
                y_test.append(cls)
            else:
                y_test.append(-1) # unknown user
            X_test.append(os.path.join(train_dir, cls, img))
    y_pred = []

    for i in range(ceil(len(X_test) / batch_size)):
        batch = X_test[
            i * batch_size:min((i + 1) * batch_size, len(X_test))
        ]
        pred_y = app.identify(batch)
        y_pred.extend(list(pred_y))
    report = classification_report(y_test, y_pred)
    print(report)





@click.command()
@click.option("--train_dir", "-t", required=True)
@click.option("--test_dir", "-e", required=True)
@click.option("--model_name", "-m", default="ArcFace")
@click.option("--backbone", "-b", default="deepface")
@click.option("--classifier", "-c", default="SVMClassifier")
@click.option("--decision_th", "-d", default=0.5)
@click.option("--bs", default=24)
def main(
    train_dir,
    test_dir,
    model_name,
    backbone,
    classifier,
    decision_th,
    bs,
):
    count_metrics(
        train_dir,
        test_dir,
        model_name,
        backbone,
        classifier,
        decision_th,
        bs
    )


main()
