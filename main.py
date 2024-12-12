# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from fuzzywuzzy import fuzz


def load_data():
    # Load and sample dataset
    df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip',
                     usecols=['question1', 'question2', 'is_duplicate'])
    df.dropna(subset=['question1', 'question2'], inplace=True)
    df = df.sample(frac=0.1, random_state=1)
    return df


def feature_engineering(df):
    # Feature Engineering
    new_df = pd.DataFrame({
        'q1_len': df['question1'].str.len(),
        'q2_len': df['question2'].str.len(),
        'q1_num_words': df['question1'].apply(lambda x: len(x.split())),
        'q2_num_words': df['question2'].apply(lambda x: len(x.split())),
        'word_count_diff': abs(
            df['question1'].apply(lambda x: len(x.split())) - df['question2'].apply(lambda x: len(x.split())))
    })

    def common_words(row):
        w1 = set(row['question1'].lower().split())
        w2 = set(row['question2'].lower().split())
        return len(w1 & w2)

    def total_words(row):
        w1 = set(row['question1'].lower().split())
        w2 = set(row['question2'].lower().split())
        return len(w1 | w2)

    new_df['word_common'] = df.apply(common_words, axis=1)
    new_df['word_total'] = df.apply(total_words, axis=1)
    new_df['word_share'] = new_df['word_common'] / new_df['word_total']
    new_df['is_duplicate'] = df['is_duplicate'].astype('int8')
    return new_df


def prepare_data(df, new_df):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf.fit(pd.Series(df['question1'].tolist() + df['question2'].tolist()))

    q1_tfidf = tfidf.transform(df['question1']).toarray()
    q2_tfidf = tfidf.transform(df['question2']).toarray()
    tfidf_features = np.hstack((q1_tfidf, q2_tfidf))

    X = np.hstack((new_df[['q1_len', 'q2_len', 'q1_num_words', 'q2_num_words', 'word_common', 'word_share',
                           'word_total']].values, tfidf_features))
    y = new_df['is_duplicate'].values

    return train_test_split(X, y, test_size=0.2, random_state=1), tfidf


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Define individual models
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8)
    log_reg_model = LogisticRegression(class_weight='balanced', max_iter=500)
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced', max_depth=10)

    metrics_results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    def evaluate_model(name, model):
        y_pred = model.predict(X_test)
        metrics_results["Model"].append(name)
        metrics_results["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics_results["Precision"].append(precision_score(y_test, y_pred))
        metrics_results["Recall"].append(recall_score(y_test, y_pred))
        metrics_results["F1 Score"].append(f1_score(y_test, y_pred))

    # Train individual models
    models = {
        "XGBoost": xgb_model,
        "Logistic Regression": log_reg_model,
        "Decision Tree": decision_tree_model
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(name, model)

    # Soft Voting Ensemble
    ensemble_model_soft = VotingClassifier(
        estimators=[("XGBoost", xgb_model), ("Logistic Regression", log_reg_model),
                    ("Decision Tree", decision_tree_model)],
        voting='soft',
        weights=[0.74, 0.16, 0.10]
    )
    ensemble_model_soft.fit(X_train, y_train)
    evaluate_model("Soft Voting Ensemble", ensemble_model_soft)

    # Hard Voting Ensemble
    ensemble_model_hard = VotingClassifier(
        estimators=[("XGBoost", xgb_model), ("Logistic Regression", log_reg_model),
                    ("Decision Tree", decision_tree_model)],
        voting='hard'
    )
    ensemble_model_hard.fit(X_train, y_train)
    evaluate_model("Hard Voting Ensemble", ensemble_model_hard)

    return pd.DataFrame(metrics_results), ensemble_model_soft


def plot_results(metrics_df):
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for model_name in metrics_df.index:
        plt.plot(metrics, metrics_df.loc[model_name, metrics], marker='o', label=model_name)

    plt.title('Performance Comparison of Models and Voting Ensembles')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def predict_similarity(question1, question2, tfidf, scaler, ensemble_model_soft):
    q1_len = len(question1)
    q2_len = len(question2)
    q1_num_words = len(question1.split())
    q2_num_words = len(question2.split())

    w1 = set(question1.lower().split())
    w2 = set(question2.lower().split())
    word_common = len(w1 & w2)
    word_total = len(w1 | w2)
    word_share = word_common / word_total if word_total != 0 else 0

    q1_tfidf = tfidf.transform([question1]).toarray()[:, :3000]
    q2_tfidf = tfidf.transform([question2]).toarray()[:, :3000]

    combined_features = np.hstack((
        np.array([[q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_share, word_total]]),
        q1_tfidf,
        q2_tfidf
    ))

    combined_features_scaled = scaler.transform(combined_features)
    prediction = ensemble_model_soft.predict(combined_features_scaled)
    return "Similar" if prediction[0] == 1 else "Not Similar"


if __name__ == "__main__":
    df = load_data()
    new_df = feature_engineering(df)
    (X_train, X_test, y_train, y_test), tfidf = prepare_data(df, new_df)
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    metrics_df, ensemble_model_soft = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Display metrics and plot results
    print(metrics_df)
    plot_results(metrics_df)

    # Test custom input
    question1 = "What's the best milk brand in Singapore?"
    question2 = "Which milk brand is the best choice in Singapore?"
    result = predict_similarity(question1, question2, tfidf, scaler, ensemble_model_soft)
    print(f"The similarity between the questions is: {result}")
