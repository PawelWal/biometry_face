from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Classifier(ABC):

    def __init__(self, decistion_th):
        self.decision_th = decistion_th
        self.classes = None

    def train(self, X, y):
        pass

    def predict_with_probs(self, x):
        pass

    def predict(self, x):
        res = self.predict_with_probs(x)
        dec = np.argmax(res, axis=1)
        final_dec = []
        for i, d in enumerate(dec):
            if res[i][d] >= self.decision_th:
                final_dec.append(d)
            else:
                final_dec.append(-1)
        if self.classes is not None:
            result = []
            for d in final_dec:
                if d == -1:
                    result.append(-1)
                else:
                    result.append(self.classes[d])
            return result
        return final_dec

    def verify_cls(self, x, y):
        pred = self.predict(x)
        y = [y] if not isinstance(y, list) else y
        return [p == c for p, c in zip(pred, y)]



class SVMClassifier(Classifier):
    def __init__(
        self,
        decistion_th=0.5,
        params={'gamma': 'auto'}):
        super().__init__(decistion_th)
        self.clf = make_pipeline(
            StandardScaler(),
            SVC(**params, probability=True)
        )

    def train(self, X, y):
        self.clf.fit(X, y)
        self.classes = self.clf.classes_

    def predict_with_probs(self, x):
        return self.clf.predict_proba(x)


class DistanseClassifier(Classifier):

    def __init__(self, decistion_th=0.9):
        super().__init__(decistion_th)

    def train(self, X, y):
        self.X = X
        self.y = y
        self.classes = y

    def predict_with_probs(self, x):
        sim = cosine_similarity(self.X, x)
        return sim.T
