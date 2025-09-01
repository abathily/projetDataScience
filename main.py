import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="ML Project", layout="wide")
st.title(" Machine Learning - Classification & Régression")

# ----------------- SESSION STATE -----------------
if "model" not in st.session_state:
    st.session_state.model = None
if "task" not in st.session_state:
    st.session_state.task = None
if "X_columns" not in st.session_state:
    st.session_state.X_columns = None

# ----------------- CHOIX DATASET -----------------
st.sidebar.header("1. Choisir un dataset")
dataset_choice = st.sidebar.selectbox(
    "Sélectionner un dataset",
    ("Iris (Classification)", "Wine (Classification)", "Diabetes (Régression)", "California Housing (Régression)", "Charger un CSV")
)

# Charger le dataset choisi
if dataset_choice == "Iris (Classification)":
    data = load_iris(as_frame=True)
    df = data.frame
    target_name = "target"
    task = "classification"

elif dataset_choice == "Wine (Classification)":
    data = load_wine(as_frame=True)
    df = data.frame
    target_name = "target"
    task = "classification"

elif dataset_choice == "Diabetes (Régression)":
    data = load_diabetes(as_frame=True)
    df = data.frame
    target_name = "target"
    task = "regression"

elif dataset_choice == "California Housing (Régression)":
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    target_name = "target"
    task = "regression"

else:
    uploaded_file = st.sidebar.file_uploader("Uploader votre CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données uploadées :")
        st.write(df.head())
        target_name = st.sidebar.selectbox("Choisir la colonne cible", df.columns)
        task = st.sidebar.radio("Type de tâche", ["classification", "regression"])
    else:
        st.warning("Veuillez uploader un fichier CSV.")
        st.stop()

# ----------------- Définir X et y -----------------
y = df[target_name]
X = df.drop(columns=[target_name])

# ----------------- TRAITEMENT DES DONNÉES -----------------
# Encodage des colonnes catégorielles
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Remplacer les valeurs manquantes par la moyenne (ou zéro pour classification)
X = X.fillna(X.mean())
if task == "regression":
    y = y.fillna(y.mean())
else:
    y = y.fillna(0)

# ----------------- EXPLORATION -----------------
st.subheader(" Exploration des données")
st.write("Dimensions :", df.shape)
st.write("Valeurs manquantes :", df.isnull().sum().sum())
st.write("Aperçu :", df.head())
st.write("Statistiques descriptives :")
st.write(df.describe())

# ----------------- VISUALISATION -----------------
st.subheader(" Visualisation")
if st.checkbox("Afficher la distribution des variables numériques"):
    fig = df.hist(figsize=(12, 8))
    st.pyplot(plt.gcf())

if st.checkbox("Afficher la heatmap des corrélations"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------- MODELISATION -----------------
st.subheader(" Modélisation")

# Split
test_size = st.slider("Taille du test set (%)", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Choix modèle
if task == "classification":
    model_choice = st.radio("Choisir un modèle", ("Logistic Regression", "Random Forest"))
    model = LogisticRegression(max_iter=1000) if model_choice == "Logistic Regression" else RandomForestClassifier()
else:
    model_choice = st.radio("Choisir un modèle", ("Linear Regression", "Random Forest"))
    model = LinearRegression() if model_choice == "Linear Regression" else RandomForestRegressor()

# Entraînement
if st.button(" Entraîner le modèle"):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Sauvegarde dans session_state
        st.session_state.model = model
        st.session_state.task = task
        st.session_state.X_columns = X.columns

        if task == "classification":
            st.success(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report :")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            st.success(f"MSE : {mean_squared_error(y_test, y_pred):.2f}")
            st.success(f"R2 Score : {r2_score(y_test, y_pred):.2f}")
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du modèle : {e}")

# ----------------- DEPLOIEMENT SIMPLE -----------------
st.subheader(" Tester une prédiction")
if st.checkbox("Faire une prédiction sur de nouvelles données"):
    if st.session_state.model is None:
        st.error("⚠ Veuillez d'abord entraîner le modèle avant de prédire.")
    else:
        input_data = []
        for col in st.session_state.X_columns:
            val = st.number_input(
                f"Entrer {col}",
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )
            input_data.append(val)

        if st.button("Prédire"):
            try:
                pred = st.session_state.model.predict([input_data])[0]
                st.success(f"Résultat de la prédiction : {pred}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
