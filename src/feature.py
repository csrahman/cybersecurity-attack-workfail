from dataset import DatasetLoader
from dataset_encode import DataEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np


class FeatureSelectorEvaluator:
    def __init__(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42,
        scoring="f1_weighted",
        handle_imbalance=True,
    ):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring
        self.handle_imbalance = handle_imbalance

        self.results = {}

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            stratify=self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def check_class_distribution(self):
        print("\n📊 Class Distribution:")
        counter = Counter(self.y)
        for label, count in counter.items():
            print(f"Class {label}: {count} samples")

        plt.bar(counter.keys(), counter.values())
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution")
        plt.show()

    def _evaluate_model(self, X_train_fs, X_test_fs):
        model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight="balanced" if self.handle_imbalance else None,
        )
        model.fit(X_train_fs, self.y_train)
        y_pred = model.predict(X_test_fs)
        return f1_score(self.y_test, y_pred, average="weighted")

    def evaluate_kbest(self, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_fs = selector.fit_transform(self.X_train, self.y_train)
        X_test_fs = selector.transform(self.X_test)
        score = self._evaluate_model(X_train_fs, X_test_fs)
        self.results["SelectKBest"] = score
        return score

    def evaluate_rfe(self, n_features=10):
        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_train_fs = selector.fit_transform(self.X_train, self.y_train)
        X_test_fs = selector.transform(self.X_test)
        score = self._evaluate_model(X_train_fs, X_test_fs)
        self.results["RFE"] = score
        return score

    def evaluate_baseline(self):
        score = self._evaluate_model(self.X_train, self.X_test)
        self.results["All Features"] = score
        return score

    def compare_all(self, k=10):
        print("Evaluating feature selection methods...\n")
        self.check_class_distribution()

        baseline = self.evaluate_baseline()
        kbest = self.evaluate_kbest(k)
        rfe = self.evaluate_rfe(k)

        print(f"\n📈 F1-Score (All Features):  {baseline:.4f}")
        print(f"📈 F1-Score (SelectKBest):   {kbest:.4f}")
        print(f"📈 F1-Score (RFE):           {rfe:.4f}")

        best_method = max(self.results, key=self.results.get)
        print(
            f"\n✅ Best Feature Selection Method: {best_method} (F1-Score: {self.results[best_method]:.4f})"
        )

        return best_method, self.results


"""
csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
loader = DatasetLoader(csv_file)
df = loader.load_data()

de = DataEncoder(target_column="Attack Type", scale_features=True)
X, y = de.encode(df)

fse = FeatureSelectorEvaluator(X, y)
best_method, all_scores = fse.compare_all(k=20)
"""
