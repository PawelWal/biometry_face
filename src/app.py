import os
import numpy as np
from .backbone import build_representation
from importlib import import_module
from time import time
from collections import defaultdict
from time import time


class FaceVer:

    def __init__(
        self,
        model_name="ArcFace",
        backbone = "deepface",
        classifier = "SVMClassifier",
        decision_th=0.5
    ):
        self.model_name = model_name
        self.backbone = backbone
        self.decision_th = decision_th
        self.backbone_instance = None
        self.classifier_name = classifier
        self.classifier = None
        self.classes = []
        self.X_rep = []
        self.y = []

    def build_representation(self, img_list):
        return build_representation(
            img_list,
            model_name=self.model_name,
            method=self.backbone
        )

    def train(self, train_dir):
        X, y = [], []
        for cls in os.listdir(train_dir):
            for img in os.listdir(os.path.join(train_dir, cls)):
                X.append(os.path.join(train_dir, cls, img))
                y.append(cls)

        print(np.array(X).shape, np.array(y).shape)
        assert np.array(X).shape[0] == np.array(y).shape[0]
        # Do shuffle???
        start = time()
        X_rep = self.build_representation(X)
        print(f"Building representation took {time() - start}")
        self.X_rep = X_rep
        self.y = y
        self.__train(X_rep, y)

    def __train(self, X_rep, y):
        clf_class = getattr(
            import_module("src.classifier"),
            self.classifier_name
        )
        if self.classifier_name == "KNNClassifier":
            classes_dict = defaultdict(int)
            for cls in y:
                classes_dict[cls] += 1
            self.classifier = clf_class(
                self.decision_th,
                min(classes_dict.values())
            )
        else:
            self.classifier = clf_class(self.decision_th)
        start = time()
        print(f"Training classifier {self.classifier_name} "
              f"with {len(X_rep)} and {len(set(y))} classes")
        self.classifier.train(X_rep, y)
        self.classes = self.classifier.classes
        print(f"Training done in {time() - start}")

    def add_user(
        self,
        user_dir,
    ):
        X, y = [], []
        for img in os.listdir(user_dir):
            X.append(os.path.join(user_dir, img))
            y.append(user_dir)

        X_rep = self.build_representation(X)
        self.X_rep.extend(X_rep)
        self.y.extend(y)
        self.__train(self.X_rep, self.y) # retrain with new user

    def verify(
        self,
        user_img,
        user_cls
    ):
        if self.classifier is None:
            raise ValueError("Classifier is not trained yet")
        user_img = [user_img] if not isinstance(user_img, list) else user_img
        user_cls = [user_cls] if not isinstance(user_cls, list) else user_cls
        if len(user_img) != len(user_cls):
            raise ValueError("Number of images and classes must be equal")
        user_rep = self.build_representation(user_img)
        return self.classifier.verify_cls(user_rep, user_cls)

    def identify(
        self,
        user_img
    ):
        """
        Return the class of the user -1 if not found
        """
        if self.classifier is None:
            raise ValueError("Classifier is not trained yet")
        user_img = [user_img] if not isinstance(user_img, list) else user_img
        user_rep = self.build_representation(user_img)
        return self.classifier.predict(user_rep)
