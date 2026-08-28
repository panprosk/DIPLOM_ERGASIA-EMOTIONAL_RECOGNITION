from sklearn.preprocessing import StandardScaler
import joblib


class FeatureScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):

        self.scaler.fit(X)

    def transform(self, X):

        return self.scaler.transform(X)

    def fit_transform(self, X):

        return self.scaler.fit_transform(X)

    def save(self, path):

        joblib.dump(self.scaler, path)

    def load(self, path):

        self.scaler = joblib.load(path)