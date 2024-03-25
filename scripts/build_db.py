from deepface import DeepFace
import click
import os


DATADIR = "/home/bwalkow/repos/biometry_face/samples/"
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


def build_db():
    train_dir = DATADIR + "train/"
    for file in os.listdir(train_dir):
        pid = file.split("person")[1].split("_")[0]
        print(file, pid)
        # for model in MODELS:
        #     print(model)
        #     DeepFace.build_model(files, model_name=model, enable_face_analysis=True)
        #     print("Model built successfully")


# @click.command()
def main():
    build_db()


main()
