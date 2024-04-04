from src import FaceVer
import click
import os
from sklearn.metrics import classification_report
from math import ceil
import matplotlib.pyplot as plt
import json


def count_metrics(
    app,
    test_dir,
    test_dir_unknown,
    batch_size=24,
):
    X_test, X_test_unknown, y_test = [], [], []
    name = test_dir.split("/")[-1]
    classifier = app.classifier_name
    print(f"Testing {name} directory...")
    for cls in os.listdir(test_dir):
        for img in os.listdir(os.path.join(test_dir, cls)):
            y_test.append(int(cls))
            X_test.append(os.path.join(test_dir, cls, img))

    for cls in os.listdir(test_dir_unknown):
        for img in os.listdir(os.path.join(test_dir_unknown, cls)):
            X_test_unknown.append(os.path.join(test_dir_unknown, cls, img))

    y_proba_unknown = []
    for i in range(ceil(len(X_test_unknown) / batch_size)):
        batch = X_test_unknown[
            i * batch_size:min((i + 1) * batch_size, len(X_test_unknown))
        ]
        pred_y, proba = app.identify(batch)
        y_proba_unknown.extend(proba)

    y_pred = []
    y_proba = []
    for i in range(ceil(len(X_test) / batch_size)):
        batch = X_test[
            i * batch_size:min((i + 1) * batch_size, len(X_test))
        ]
        pred_y, proba = app.identify(batch)
        y_pred.extend(pred_y)
        y_proba.extend(proba)

    # far calculation for impostors & frr calculation for genuine
    right_indexes = [i for i, (x, y) in enumerate(zip(y_test, y_pred)) if x == y]
    far = []
    frr = []
    threshold = []
    for cur_threshold in range(100):
        num_far = 0
        num_frr = 0
        for prob in y_proba_unknown:
            if prob * 100 > cur_threshold:
                num_far += 1
        for idx in right_indexes:
            if y_proba[idx] * 100 < cur_threshold:
                num_frr += 1
        far.append(num_far / len(y_proba_unknown))
        frr.append(num_frr / len(y_test))
        threshold.append(cur_threshold / 100)

    fig, ax = plt.subplots()
    ax.plot(threshold, far, 'r--', label='FAR')
    ax.plot(threshold, frr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Data subset')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.savefig(f"metrics/metrics_{classifier}_{name}.png")

    # print("True", y_test[:10])
    # print("Pred", y_pred[:10])
    report = classification_report(y_test, y_pred)
    # print(report)
    with open(f"metrics/report_{classifier}_{name}.txt", "w") as f:
        f.write(report)

    cross_point = 0
    for i, (x, y) in enumerate(zip(far, frr)):
        if x <= y:
            cross_point = i
            break
    res_dict = {
        "name": f"{classifier}_{name}",
        "far": far,
        "frr": frr,
        "cross_point": cross_point,
        "threshold": threshold,
    }
    with open(f"metrics/metrics.jsonl", "a") as f:
        f.write(json.dumps(res_dict) + "\n")


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
