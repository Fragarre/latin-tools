import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import cdist
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

def clear_test_folder():
    data_path = './new_data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

clear_data_folder()
clear_test_folder()

# --- Funciones para extracción de archivos ZIP --- #
def unzip_data(zip_file, extract_to='./data/'):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    st.success("¡Datos extraídos correctamente!. (Espera mientras el estado sea RUNNING...)")

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
        title='Selección de n_components para TruncatedSVD',
        xaxis_title='Número de componentes',
        yaxis_title='Varianza acumulada explicada',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    return svd, var_acumulada, optimal_n

# --- Cargar textos y entrenar modelo con split --- #
def load_and_train_model_split(ngram_min, ngram_max, test_size=0.2):
    corpus_path = './data/'
    texts, labels, filenames = [], [], []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.txt'):
            full_path = os.path.join(corpus_path, filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                author = filename.split("_")[0] if "_" in filename else "Desconocido"
                labels.append(author)
                filenames.append(filename[:-4])  # sin .txt

    if len(texts) < 2:
        st.error("Se necesitan al menos 2 archivos para entrenar el modelo.")
        return None

    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        texts, labels, filenames, test_size=test_size, stratify=labels, random_state=42
    )

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X_vec_train = vectorizer.fit_transform(X_train)
    X_vec_test = vectorizer.transform(X_test)
    st.subheader("Análisis de número de componentes para algoritmo")
    svd_model, varianza, mejor_n = plot_svd_variance_streamlit(X_vec_train, max_components=150, threshold=0.90)
    
    st.success(f"Para explicar al menos el 90% de la varianza, se toman {mejor_n} componentes.")
    svd = TruncatedSVD(n_components=mejor_n, random_state=42)
    X_reduced_train = svd.fit_transform(X_vec_train)
    X_reduced_test = svd.transform(X_vec_test)

    clf = NearestCentroid()
    clf.fit(X_reduced_train, y_train)
    y_pred = clf.predict(X_reduced_test)

    return {
        "clf": clf,
        "vectorizer": vectorizer,
        "svd": svd,
        "X_train_reduced": X_reduced_train,
        "y_train": y_train,
        "filenames_train": filenames_train,
        "X_test_reduced": X_reduced_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "filenames_test": filenames_test,
    }

# --- Calcular distancias y probabilidades de autoría --- #
def calculate_authorship_probabilities(model_result, new_zip):
    temp_path = './new_data/'
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    unzip_data(new_zip, extract_to=temp_path)

    clf = model_result['clf']
    vectorizer = model_result['vectorizer']
    svd = model_result['svd']

    new_texts, new_filenames = [], []
    for filename in os.listdir(temp_path):
        if filename.endswith('.txt'):
            with open(os.path.join(temp_path, filename), 'r', encoding='utf-8') as f:
                new_texts.append(f.read())
                new_filenames.append(filename[:-4])

    if not new_texts:
        st.warning("No se encontraron archivos .txt en el nuevo ZIP.")
        return

    X_new = vectorizer.transform(new_texts)
    X_new_reduced = svd.transform(X_new)

    distances = cdist(X_new_reduced, clf.centroids_, metric='euclidean')

    # --- Tabla de distancias --- #
    distance_table = pd.DataFrame(distances, columns=clf.classes_)
    distance_table.insert(0, "Texto", new_filenames)

    st.write("### Tabla de distancias a centroides")
    st.dataframe(
        distance_table.style.highlight_min(subset=clf.classes_, axis=1, color='lightblue'),
        use_container_width=True
    )

    # --- Tabla de probabilidades con softmax inversa --- #
    scaling_factor = 10  # Puedes probar con 5, 10, 20
    inv_distances = np.exp(-scaling_factor * distances)
    probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)
    probs_percent = probs * 100

    prob_table = pd.DataFrame(probs_percent, columns=clf.classes_)
    prob_table.insert(0, "Texto", new_filenames)

    st.write("### Tabla de probabilidades de autoría (%)")
    st.dataframe(
        prob_table.style.format({col: "{:.1f}%" for col in clf.classes_})
                   .highlight_max(subset=clf.classes_, axis=1, color='lightgreen'),
        use_container_width=True
    )

    # --- Gráfica tipo "código de barras" por texto --- #
    st.write("### Visualización de probabilidades por texto")
    for i, text_name in enumerate(new_filenames):
        fig = px.bar(
            x=clf.classes_,
            y=probs_percent[i],
            labels={'x': 'Autor', 'y': 'Probabilidad (%)'},
            title=f"Probabilidades de autoría para '{text_name}'",
            text=[f"{p:.1f}%" for p in probs_percent[i]],
        )
        fig.update_traces(marker_color='lightblue', textposition='outside')
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)



# --- Interfaz Streamlit --- #
st.subheader("Análisis autoría. (Recargar página para nuevo análisis)")

st.sidebar.markdown("""
### Instrucciones
1. Sube un archivo `.zip` con textos etiquetados para entrenar.
2. Sube un segundo `.zip` con textos a clasificar.
3. Cada archivo debe empezar con el nombre del autor seguido de guion bajo (ej. `Seneca_DeBeneficiis.txt`).
""")

st.sidebar.markdown("### Configuración de n-gramas")
ngram_min = st.sidebar.number_input("n-grama mínimo", min_value=1, max_value=10, value=2)
ngram_max = st.sidebar.number_input("n-grama máximo", min_value=1, max_value=10, value=4)

zip_train = st.file_uploader("Sube archivo .zip de entrenamiento", type="zip")
zip_predict = st.file_uploader("Sube archivo .zip con textos a clasificar", type="zip")

if zip_train:
    unzip_data(zip_train)
    result = load_and_train_model_split(ngram_min, ngram_max)

    if result:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        import numpy as np

        clf = result['clf']
        y_test = result['y_test']
        y_pred = result['y_pred']
        filenames_test = result['filenames_test']

        st.write("### Evaluación del modelo")
        accuracy = np.mean(np.array(y_test) == np.array(y_pred))
        st.write(f"**Precisión en test:** {accuracy:.2%}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

        # Crear el plot y personalizar
        fig, ax = plt.subplots(figsize=(8, 6))  # Tamaño ajustable si lo necesitas
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

        # Ajustar etiquetas
        ax.set_xlabel("Etiqueta predicha", fontsize=10)
        ax.set_ylabel("Etiqueta real", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)  # Reducir tamaño de los ticks

        st.pyplot(fig, use_container_width=True)


        errors = []
        for real, pred, filename in zip(y_test, y_pred, filenames_test):
            if real != pred:
                errors.append({
                    "Texto": filename,
                    "Autor real": real,
                    "Autor predicho": pred
                })

        if errors:
            st.write("### Textos mal clasificados en el conjunto de test")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)
        else:
            st.success("✅ Todos los textos del conjunto de test fueron clasificados correctamente.")

        if zip_predict:
            calculate_authorship_probabilities(result, zip_predict)

    else:
        st.warning("No se pudo entrenar el modelo. Verifica los archivos.")
