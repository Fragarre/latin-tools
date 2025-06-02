import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from umap import UMAP
import plotly.express as px
import pandas as pd
import shutil
import plotly.graph_objects as go

# --- Borrar automáticamente ./data al iniciar la app --- #
def clear_data_folder():
    data_path = './data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

clear_data_folder()

# --- funcion para ver cambios en varianza ---
def plot_svd_variance_streamlit(X, max_components=200, threshold=0.90, random_state=42):
    """
    Aplica TruncatedSVD a X, genera un gráfico interactivo en Streamlit y devuelve el modelo y el número óptimo de componentes.

    Parámetros:
        X               - matriz dispersa o densa (salida de TfidfVectorizer)
        max_components  - número máximo de componentes a considerar
        threshold       - proporción de varianza explicada deseada (ej. 0.90 para 90%)
        random_state    - semilla para reproducibilidad

    Devuelve:
        svd             - modelo TruncatedSVD ajustado
        var_acumulada   - array de varianza acumulada
        optimal_n       - número mínimo de componentes que alcanzan el threshold
    """
    # Entrenamiento SVD
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    var_acumulada = np.cumsum(svd.explained_variance_ratio_)

    # Cálculo del n óptimo
    optimal_n = np.argmax(var_acumulada >= threshold) + 1

    # Gráfico Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, max_components + 1)),
        y=var_acumulada,
        mode='lines+markers',
        name='Varianza acumulada'
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{int(threshold * 100)}% varianza explicada",
        annotation_position="bottom right"
    )

    fig.add_vline(
        x=optimal_n,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"n_components = {optimal_n}",
        annotation_position="top left"
    )

    fig.update_layout(
        title='Selección de n_components para algoritmo',
        xaxis_title='Número de componentes',
        yaxis_title='Varianza acumulada explicada',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    return svd, var_acumulada, optimal_n

st.subheader("Análisis estilométrico (Recargar página para nuevo análisis)")
# --- Función para extraer el archivo ZIP --- #
def unzip_data(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('./data/')
    st.success("¡Datos extraídos correctamente!. (Espera mientras el estado sea RUNNING...)")

# --- Función para cargar los textos y generar el modelo --- #
def load_and_train_model(ngram_min, ngram_max):
    corpus_path = './data/'
    texts, labels, filenames = [], [], []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.txt'):
            full_path = os.path.join(corpus_path, filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

                if "_" in filename:
                    author = filename.split("_")[0]
                else:
                    author = "Desconocido"

                labels.append(author)
                filenames.append(filename[:-4])  # sin extensión

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)

    # # Obtener los 500 n-gramas con mayor importancia (promedio de TF-IDF)
    # tfidf_means = np.asarray(X.mean(axis=0)).flatten()
    # feature_names = vectorizer.get_feature_names_out()
    # top_n = 500
    # top_indices = np.argsort(tfidf_means)[::-1][:top_n]
    # top_features = [(feature_names[i], tfidf_means[i]) for i in top_indices]

    # # Mostrar en Streamlit
    # st.subheader(f"Top {top_n} n-gramas por importancia media (TF-IDF)")
    # df_top = pd.DataFrame(top_features, columns=["n-grama", "TF-IDF medio"])
    # st.dataframe(df_top)

    # Agregar st.write para mostrar el total de n-gramas generados
    st.write("Total de n-gramas generados:", len(vectorizer.get_feature_names_out()))

    st.subheader("Análisis de número de componentes")
    svd_model, varianza, mejor_n = plot_svd_variance_streamlit(X, max_components=150, threshold=0.90)

    st.success(f"Para explicar al menos el 90% de la varianza, se toman {mejor_n} componentes.")
    svd = TruncatedSVD(n_components=mejor_n, random_state=42)
    X_reduced = svd.fit_transform(X)

    clf = NearestCentroid()
    clf.fit(X_reduced, labels)

    return clf, vectorizer, svd, texts, labels, filenames, X_reduced

# --- Identificar errores en la matriz de confusión --- #
def identify_confusion_errors(labels, y_pred, filenames, clf):
    errors = []
    cm = confusion_matrix(labels, y_pred, labels=clf.classes_)
    for i in range(len(clf.classes_)):
        for j in range(len(clf.classes_)):
            if i != j and cm[i, j] > 0:
                misclassified_files = [filenames[idx] for idx, label in enumerate(labels) 
                                       if label == clf.classes_[i] and y_pred[idx] == clf.classes_[j]]
                errors.append({
                    'true_label': clf.classes_[i],
                    'predicted_label': clf.classes_[j],
                    'misclassified_files': misclassified_files
                })
    return errors

# --- Interfaz de usuario en Streamlit --- #
st.write("Sube un archivo .zip con los datos de los autores para entrenar el modelo.")
st.markdown("""
### Instrucciones

Sube un archivo `data.zip` que contenga los archivos .txt directamente (sin subcarpetas).  
Cada archivo debe comenzar con el nombre del autor, seguido de "_", por ejemplo: `Seneca_DeBeneficiis.txt`.
""")

ngram_min = st.sidebar.number_input("n-grama mínimo", min_value=1, max_value=10, value=2)
ngram_max = st.sidebar.number_input("n-grama máximo", min_value=1, max_value=10, value=4)

point_size = st.sidebar.number_input("Tamaño del punto en los gráficos", min_value=1, max_value=50, value=10)
uploaded_zip = st.file_uploader("Sube un archivo .zip", type=["zip"])

if uploaded_zip is not None:
    unzip_data(uploaded_zip)
    clf, vectorizer, svd, texts, labels, filenames, X_reduced = load_and_train_model(ngram_min, ngram_max)

    st.write("Generando la matriz de confusión...")
    # Predicción y matriz de confusión
    y_pred = clf.predict(X_reduced)
    cm = confusion_matrix(labels, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

    # Crear el plot y personalizar
    fig, ax = plt.subplots(figsize=(8, 6))  # Puedes ajustar el tamaño según tus necesidades
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

    # Etiquetas personalizadas en español
    ax.set_xlabel("Etiqueta predicha", fontsize=10)
    ax.set_ylabel("Etiqueta real", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)  # Tamaño más pequeño para las etiquetas

    # Mostrar en Streamlit
    st.pyplot(fig, use_container_width=True)

    st.write("Generando visualizaciones...")
    n_samples = len(X_reduced)

    # --- t-SNE con ajuste dinámico de perplexity --- #
    perplexity = min(30, max(2, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_reduced)

    tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
    tsne_df['author'] = labels
    tsne_df['filename'] = filenames
    fig_tsne = px.scatter(tsne_df, x='x', y='y', color='author', hover_data=['filename'],
                          title="Visualización t-SNE", width=900, height=600)
    fig_tsne.update_traces(marker=dict(size=point_size))
    st.plotly_chart(fig_tsne)

    # --- UMAP con ajuste dinámico de n_neighbors --- #
    n_neighbors = min(15, max(2, n_samples // 3))
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    X_umap = reducer.fit_transform(X_reduced)

    umap_df = pd.DataFrame(X_umap, columns=['x', 'y'])
    umap_df['author'] = labels
    umap_df['filename'] = filenames
    fig_umap = px.scatter(umap_df, x='x', y='y', color='author', hover_data=['filename'],
                          title="Visualización UMAP", width=900, height=600)
    fig_umap.update_traces(marker=dict(size=point_size))
    st.plotly_chart(fig_umap)

    # --- Tabla con distancias a centroides --- #
    distances = cdist(X_reduced, clf.centroids_, metric='euclidean')
    clf_classes = clf.classes_

    # Crear la tabla con distancias
    distance_matrix = []
    for i, fname in enumerate(filenames):
        row = {"Texto": fname, "Autor": labels[i]}
        for j, author in enumerate(clf_classes):
            row[f"Distancia_{author}"] = round(distances[i][j], 5)
        closest_author = clf_classes[np.argmin(distances[i])]
        row["Más cercano"] = closest_author
        distance_matrix.append(row)

    # Crear DataFrame
    df_distances = pd.DataFrame(distance_matrix)

    # Reordenar columnas: Texto, Autor, Autor más cercano, luego distancias
    cols = ["Texto", "Autor", "Más cercano"] + [col for col in df_distances.columns if col.startswith("Distancia")]
    df_distances = df_distances[cols]

    # Estilo para resaltar en verde el mínimo de cada fila (entre distancias)
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: lightgreen' if v else '' for v in is_min]

    # Extraer solo columnas de distancia
    dist_cols = [col for col in df_distances.columns if col.startswith("Distancia")]
    styled_df = df_distances.style.apply(highlight_min, subset=dist_cols, axis=1)

    # Mostrar en Streamlit
    st.write("### Distancia de cada texto a los centroides de autor")
    st.dataframe(styled_df, use_container_width=True)