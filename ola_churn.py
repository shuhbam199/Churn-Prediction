"""Object-oriented training pipeline for Ola driver churn prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import pickle


class OlaChurnModel:
    """Pipeline for training the Ola driver churn model."""

    def __init__(self, data_path: str = "ola.csv") -> None:
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.model: xgb.XGBClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None

    def load_data(self) -> pd.DataFrame:
        """Load the raw CSV file."""
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess(self) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess raw data and create features."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data first.")

        df = self.df.copy()
        df['City_encoded'] = df.groupby('City')['Grade'].transform('mean').round(3)
        df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'])
        df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')
        df['MMM-YY'] = pd.to_datetime(df['MMM-YY'])

        df2 = df.drop(columns=['City', 'Unnamed: 0'])
        knn = KNNImputer(n_neighbors=5, metric='nan_euclidean', weights='distance')
        num = pd.DataFrame(
            knn.fit_transform(df2.select_dtypes(np.number)),
            columns=df2.select_dtypes(np.number).columns,
        )
        cat = df2.select_dtypes(exclude=[np.number])
        df_new = pd.concat([cat, num], axis=1)
        df_new['Churn'] = df_new['LastWorkingDate'].notna().map({True: 1, False: 0})

        df_new['Last_grade'] = df_new.groupby('Driver_ID')['Grade'].transform('last')
        df_new['first_grade'] = df_new.groupby('Driver_ID')['Grade'].transform('first')
        df_new['Last_rat'] = df_new.groupby('Driver_ID')['Quarterly Rating'].transform('last')
        df_new['first_rat'] = df_new.groupby('Driver_ID')['Quarterly Rating'].transform('first')
        df_new['Last_income'] = df_new.groupby('Driver_ID')['Income'].transform('last')
        df_new['first_income'] = df_new.groupby('Driver_ID')['Income'].transform('first')

        df_new['Income_increased'] = np.where(
            df_new['Last_income'] - df_new['first_income'] > 0, 1, 0
        )
        df_new['Grade_improved'] = np.where(
            df_new['Last_grade'] - df_new['first_grade'] > 0,
            1,
            np.where(
                df_new['Last_grade'] - df_new['first_grade'] < 0,
                -1,
                0,
            ),
        )
        df_new['Rating_change'] = df_new['Last_rat'] - df_new['first_rat']
        df_new.drop(
            columns=['first_grade', 'first_rat', 'Last_income', 'first_income'],
            inplace=True,
        )

        df_group = df_new.groupby('Driver_ID').agg(
            {
                'MMM-YY': 'max',
                'Dateofjoining': 'max',
                'LastWorkingDate': 'max',
                'Age': 'max',
                'Gender': 'max',
                'Income': 'sum',
                'Joining Designation': 'max',
                'Total Business Value': 'sum',
                'Education_Level': 'max',
                'City_encoded': 'max',
                'Churn': 'max',
                'Last_grade': 'max',
                'Income_increased': 'max',
                'Last_rat': 'max',
                'Grade_improved': 'max',
                'Rating_change': 'max',
            }
        ).reset_index()

        df_group['Time_served'] = np.where(
            df_group['LastWorkingDate'].isna(),
            (df_group['MMM-YY'] - df_group['Dateofjoining']).abs(),
            df_group['LastWorkingDate'] - df_group['Dateofjoining'],
        )
        df_group['Time_served'] = df_group['Time_served'].dt.days
        df_group['Gender'] = df_group['Gender'].round().astype(int)

        df_g = df_group.drop(columns=['MMM-YY', 'Dateofjoining', 'LastWorkingDate'])
        X = df_g.drop(columns=['Driver_ID', 'City_encoded', 'Time_served', 'Churn'])
        y = df_g['Churn']

        self.X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        self.y = y
        return self.X, self.y

    def train(self) -> tuple[str, pd.Series]:
        """Train the XGBoost model and compute evaluation metrics.

        Returns:
            tuple[str, pd.Series]: classification report and top SHAP features.
        """
        if self.X is None or self.y is None:
            raise ValueError("Preprocess data before training.")

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            reg_alpha=0.01,
            reg_lambda=0.01,
            use_label_encoder=False,
            eval_metric='logloss',
        )
        self.model.fit(X_res, y_res)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)

        self.explainer = shap.TreeExplainer(self.model)
        X_test_df = pd.DataFrame(X_test, columns=self.X.columns)
        shap_values = self.explainer.shap_values(X_test_df)
        top_feats = (
            pd.Series(shap_values[0], index=X_test_df.columns)
            .abs()
            .sort_values(ascending=False)
            .head(3)
        )
        return report, top_feats

    def save_artifacts(self) -> None:
        """Persist model, scaler and SHAP explainer."""
        if self.model is None or self.explainer is None:
            raise ValueError("Train the model before saving artifacts.")

        with open('xgb_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('xgb_explainer.pkl', 'wb') as f:
            pickle.dump(self.explainer, f)

    def run(self) -> None:
        """Execute the full pipeline."""
        self.load_data()
        self.preprocess()
        report, top_feats = self.train()
        print(report)
        print(top_feats)
        self.save_artifacts()


if __name__ == '__main__':
    OlaChurnModel().run()
