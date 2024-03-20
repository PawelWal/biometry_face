from deepface import DeepFace
import click

DATADIR = "/home/bwalkow/repos/biomerty/samples/"
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

def run_verify(
    model_nbr,
    metric
):
    model_name = MODELS[model_nbr]
    print(f"MODEL NAME {model_name}")

    result = DeepFace.verify(
        img1_path=f"{DATADIR}/img1.jpg",
        img2_path=f"{DATADIR}/img2.jpg",
        model_name=model_name,
        distance_metric=metric,
    )
    print(result)
    print("Is verified: ", result["verified"])

    result = DeepFace.verify(
        img1_path=f"{DATADIR}/img2.jpg",
        img2_path=f"{DATADIR}/img22.jpg",
        model_name=model_name,
        distance_metric=metric,
    )
    print(result)
    print("Is verified: ", result["verified"])


@click.command()
@click.option("--model_nbr", "-mn", default=4, help="Model number to use")
@click.option("--metric", "-m", default="cosine", type=click.Choice(METRICS) , help="Metric to use")
def main(
    model_nbr,
    metric,
):
    run_verify(int(model_nbr), metric)


main()
