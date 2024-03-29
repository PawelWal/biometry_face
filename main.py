from src.classifier import SVMClassifier, DistanseClassifier, KNNClassifier
# from src.app import FaceVer
import numpy as np
import click


def test():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [3, 1], [3, 1]])
    y = np.array([1, 1, 2, 2, 3, 3])
    # clf = SVMClassifier()
    # clf = DistanseClassifier()
    clf = KNNClassifier()
    clf.train(X, y)

    print(clf.predict(X))
    print(clf.predict([[-0.8, -1]]))
    print(clf.predict([[-0.8, -1], [-0.8, -1]]))
    print(clf.verify_cls([[-0.8, -1]], 0))
    print(clf.verify_cls([[-0.8, -1], [-0.8, -1]], [0, 1]))
    print(clf.predict([[0, 0]]))


# def test1(
#     train_dir,

# ):
#     app = FaceVer()
#     app.train("data/train")

def main():
    test()

main()
