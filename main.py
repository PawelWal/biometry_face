from src.classifier import SVMClassifier, DistanseClassifier
import numpy as np


def main():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [3, 1], [3, 1]])
    y = np.array([1, 1, 2, 2, 3, 3])
    # clf = SVMClassifier()
    clf = DistanseClassifier()
    clf.train(X, y)

    print(clf.predict(X))
    print(clf.predict([[-0.8, -1]]))
    print(clf.predict([[-0.8, -1], [-0.8, -1]]))
    print(clf.verify_cls([[-0.8, -1]], 0))
    print(clf.verify_cls([[-0.8, -1], [-0.8, -1]], [0, 1]))
    print(clf.predict([[0, 0]]))


main()
