from sklearn.base import BaseEstimator, TransformerMixin

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.top_symptoms = None
        self.feature_names = None

    def fit(self, X, y=None):
        all_symptoms = X.iloc[:, 0].explode().dropna()
        symptom_counts = all_symptoms.value_counts()
        self.top_symptoms = symptom_counts.head(10).index.tolist()

        # Fit only on top symptoms
        X_filtered = X.iloc[:, 0].apply(lambda x: [s for s in x if s in self.top_symptoms])
        self.mlb.fit(X_filtered)

        # Save feature names for later use
        self.feature_names = [f"symptom_{s}" for s in self.mlb.classes_]
        return self

    def transform(self, X):
        X_filtered = X.iloc[:, 0].apply(lambda x: [s for s in x if s in self.top_symptoms])
        return self.mlb.transform(X_filtered)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)



