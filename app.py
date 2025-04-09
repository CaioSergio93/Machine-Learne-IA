import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Geração de Dados Sintéticos com IA Generativa

def generate_classification_data(n_samples=500, n_features=5, n_classes=3):
    """Gera dados sintéticos para classificação"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features-1,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='Target')

def generate_clustering_data(n_samples=500, n_features=3, n_clusters=4):
    """Gera dados sintéticos para clusterização"""
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=42
    )
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names)

# Interface Streamlit

def main():
    st.title("Projeto de Machine Learning com IA Generativa")
    st.markdown("""
    ## Classificação e Clusterização com Dados Sintéticos
    Este projeto demonstra:
    - Geração automática de dados com IA Generativa
    - Modelos de classificação supervisionada
    - Técnicas de clusterização não supervisionada
    """)
    
    # Sidebar controls
    st.sidebar.header("Configurações")
    problem_type = st.sidebar.selectbox("Tipo de Problema", ["Classificação", "Clusterização"])
    
    if problem_type == "Classificação":
        classification_app()
    else:
        clustering_app()

def classification_app():
    st.header("Classificação com Dados Sintéticos")
    
    # Generate data
    n_samples = st.slider("Número de Amostras", 100, 1000, 500)
    n_features = st.slider("Número de Features", 2, 10, 5)
    n_classes = st.slider("Número de Classes", 2, 5, 3)
    
    X, y = generate_classification_data(n_samples, n_features, n_classes)
    
    # Show data
    st.subheader("Visualização dos Dados")
    st.write(f"Dimensões: {X.shape[0]} amostras, {X.shape[1]} features")
    st.dataframe(pd.concat([X, y], axis=1).head())
    
    # Plot
    if n_features >= 2:
        fig = px.scatter(
            x=X.iloc[:, 0], 
            y=X.iloc[:, 1], 
            color=y,
            title="Visualização 2D dos Dados (Primeiras 2 Features)"
        )
        st.plotly_chart(fig)
    
    # Train model
    st.subheader("Treinamento do Modelo")
    test_size = st.slider("Tamanho do Conjunto de Teste (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Results
    st.subheader("Resultados")
    st.text(classification_report(y_test, y_pred))
    
    # Feature importance
    st.subheader("Importância das Features")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)

def clustering_app():
    st.header("Clusterização com Dados Sintéticos")
    
    # Generate data
    n_samples = st.slider("Número de Amostras", 100, 1000, 500)
    n_features = st.slider("Número de Features", 2, 5, 3)
    n_clusters = st.slider("Número de Clusters", 2, 6, 4)
    
    X = generate_clustering_data(n_samples, n_features, n_clusters)
    
    # Show data
    st.subheader("Visualização dos Dados")
    st.write(f"Dimensões: {X.shape[0]} amostras, {X.shape[1]} features")
    st.dataframe(X.head())
    
    # Plot
    if n_features >= 2:
        fig = px.scatter(
            x=X.iloc[:, 0], 
            y=X.iloc[:, 1], 
            title="Visualização 2D dos Dados (Primeiras 2 Features)"
        )
        st.plotly_chart(fig)
    
    # Cluster analysis
    st.subheader("Análise de Clusters")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Plot clusters
    if n_features >= 2:
        fig = px.scatter(
            x=X.iloc[:, 0], 
            y=X.iloc[:, 1], 
            color=clusters,
            title="Resultado da Clusterização"
        )
        st.plotly_chart(fig)
    
    # Metrics
    silhouette = silhouette_score(X, clusters)
    st.metric("Silhouette Score", f"{silhouette:.2f}")
    
    st.write("""
    **Interpretação do Silhouette Score:**
    - Valores próximos a 1 indicam clusters bem definidos
    - Valores próximos a 0 indicam clusters sobrepostos
    - Valores negativos indicam amostras possivelmente atribuídas ao cluster errado
    """)

if __name__ == "__main__":
    main()
