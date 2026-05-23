from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class IrisModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, X_train, y_train):
        # fit method is used to train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test):
        if not self.is_trained:
            raise RuntimeError("Train the model before calling predict().")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return {
            "accuracy": accuracy,
            "classification_report": report,
        }