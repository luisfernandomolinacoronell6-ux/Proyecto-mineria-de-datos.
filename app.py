import streamlit as st
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Predicción de Diabetes", layout="wide")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42

st.title("Proyecto de Minería de Datos: Predicción de Diabetes")

@st.cache_data
def cargar_datos():
    RUTAS = ["diabetes.csv", "data/diabetes.csv", "/content/diabetes.csv"]
    ruta = next((r for r in RUTAS if os.path.exists(r)), None)

    if ruta is not None:
        df = pd.read_csv(ruta)
        origen = f"Archivo local usado: {ruta}"
    else:
        url_espejo = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columnas_pima = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
        df = pd.read_csv(url_espejo, names=columnas_pima)
        origen = "Archivo descargado desde el espejo de GitHub"
    return df, origen

try:
    df_raw, mensaje_origen = cargar_datos()
    st.sidebar.text(mensaje_origen)
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Datos y Limpieza", "Análisis Exploratorio", "Modelado y Evaluación"])

with tab1:
    st.header("1. Preparación de los Datos")
    
    col1, col2 = st.columns(2)
    col1.metric("Filas originales", df_raw.shape[0])
    col2.metric("Columnas originales", df_raw.shape[1])
    
    st.subheader("Vista previa de los datos")
    st.dataframe(df_raw.head())

    st.subheader("Resumen Estadístico")
    st.dataframe(df_raw.describe().T.round(2))

    COLS_CERO_INVALIDO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    ceros = pd.DataFrame({
        "ceros": (df_raw[COLS_CERO_INVALIDO] == 0).sum(),
        "% del total": ((df_raw[COLS_CERO_INVALIDO] == 0).mean() * 100).round(1),
    })
    
    st.text("Análisis de ceros:")
    st.dataframe(ceros)

    df = df_raw.drop_duplicates().copy()
    df[COLS_CERO_INVALIDO] = df[COLS_CERO_INVALIDO].replace(0, np.nan)
    
    st.text("Faltantes tras conversión:")
    st.dataframe(pd.DataFrame({"faltantes": df.isna().sum(), "%": (df.isna().mean()*100).round(1)}).T)

    def resumen_outliers(data):
        filas = {}
        for c in data.columns.drop("Outcome"):
            s = data[c].dropna()
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            n = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            filas[c] = {"atípicos": int(n), "% de válidos": round(n/len(s)*100, 1)}
        return pd.DataFrame(filas).T
    
    st.text("Resumen de valores atípicos:")
    st.dataframe(resumen_outliers(df))

with tab2:
    st.header("2. Análisis Exploratorio de Datos")
    
    st.subheader("Distribución de la Variable Objetivo")
    conteo = df["Outcome"].value_counts().sort_index()
    pct = (conteo / conteo.sum() * 100).round(1)
    
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=df, x="Outcome", ax=ax1, palette=["#4C78A8", "#E45756"])
    ax1.set_xticklabels(["Sin diabetes (0)", "Con diabetes (1)"])
    ax1.set_xlabel("")
    ax1.set_ylabel("Número de registros")
    for i, v in enumerate(conteo):
        ax1.text(i, v + 5, f"{v} ({pct[i]}%)", ha="center")
    st.pyplot(fig1)

    st.subheader("Histogramas de las Variables Predictoras")
    FEATURES = [c for c in df.columns if c != "Outcome"]
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes2.ravel(), FEATURES):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#4C78A8")
        ax.set_title(c)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Variables Predictoras según la Variable Objetivo")
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    for ax, c in zip(axes3.ravel(), FEATURES):
        sns.boxplot(data=df, x="Outcome", y=c, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.set_xticklabels(["Sin", "Con"])
        ax.set_xlabel("Diabetes")
        ax.set_title(c)
    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("Matriz de Correlación de Pearson")
    corr = df.corr(numeric_only=True)
    fig4, ax4 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax4)
    st.pyplot(fig4)

with tab3:
    st.header("3. Entrenamiento y Evaluación de Modelos")
    
    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    
    st.text(f"Dimensiones de entrenamiento: {X_train.shape[0]} | Dimensiones de prueba: {X_test.shape[0]}")

    modelos = {
        "Línea base (mayoritaria)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Regresión logística": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("esc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "Árbol de decisión": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED))]),
    }

    st.subheader("Resultados de Validación Cruzada (Entrenamiento)")
    st.text("Reporte de media y variación usando 5 particiones:")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultados_cv = []

    for nombre, pipe in modelos.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['accuracy', 'recall', 'f1'])
        
        resultados_cv.append({
            "Modelo": nombre,
            "Accuracy (Media ± Var)": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
            "Recall (Media ± Var)": f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
            "F1 (Media ± Var)": f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}"
        })
        
        pipe.fit(X_train, y_train)

    st.dataframe(pd.DataFrame(resultados_cv).set_index("Modelo"))

    st.subheader("Resultados Finales en Conjunto de Prueba (Una sola vez)")
    pred_test  = {n: m.predict(X_test) for n, m in modelos.items()}
    proba_test = {n: m.predict_proba(X_test)[:, 1] for n, m in modelos.items()}

    def metricas(y_real, y_pred, y_proba):
        return {"Accuracy": accuracy_score(y_real, y_pred),
                "Precision": precision_score(y_real, y_pred, zero_division=0),
                "Recall": recall_score(y_real, y_pred, zero_division=0),
                "F1": f1_score(y_real, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_real, y_proba)}

    tabla_test = pd.DataFrame({n: metricas(y_test, pred_test[n], proba_test[n]) for n in modelos}).T.round(3)
    st.dataframe(tabla_test)
