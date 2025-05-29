import streamlit as st
from pathlib import Path
import os
import zipfile
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Configuración inicial de Streamlit
st.subheader("Extraer textos")

# Estado de sesión para controlar el flujo
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "urls" not in st.session_state:
    st.session_state.urls = []
if "output_folder" not in st.session_state:
    st.session_state.output_folder = "output"
if "interrumpido" not in st.session_state:
    st.session_state.interrumpido = False
if "scraping" not in st.session_state:
    st.session_state.scraping = False
if "file_counter" not in st.session_state:
    st.session_state.file_counter = 1  # Nueva variable para controlar la numeración de archivos

# Crear carpeta de salida si no existe
os.makedirs(st.session_state.output_folder, exist_ok=True)

# Configuración de Selenium con Chrome (no headless)
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except WebDriverException as e:
        st.error(f"Error al iniciar el driver de Chrome: {e}")
        return None

# Función para extraer texto desde una URL
def extract_text_from_url(driver, url):
    try:
        driver.get(url)
        time.sleep(1)
        body = driver.find_element("tag name", "body")
        text = body.text.strip()

        # Frases típicas de página no encontrada
        error_indicators = [
            "Pagina amissa", 
            "Quam quaeris abest", 
            "Si epistulam mittes huic",
            "404",
            "Not Found",
            "Page not found"
        ]

        if not text or any(err in text for err in error_indicators):
            return None  # Consideramos la página como inválida

        return text
    except (WebDriverException, TimeoutException):
        return None

# Formulario de entrada de datos si no ha comenzado el scraping
if not st.session_state.scraping:
    autor = st.text_input("Nombre del autor (sin espacios ni tildes)")
    obra = st.text_input("Nombre de la obra (sin espacios ni tildes)")
    num_partes = st.number_input("Número total de capítulos/libros/partes", min_value=1, step=1)
    url_inicial = st.text_input("URL del primer capítulo")

    if st.button("Iniciar scraping"):
        # Generar todas las URLs basadas en patrón
        urls = []
        for i in range(1, int(num_partes)+1):
            nueva_url = url_inicial.replace("1", str(i), 1)
            urls.append(nueva_url)

        st.session_state.urls = urls
        st.session_state.autor = autor.lower()
        st.session_state.obra = obra.lower()
        st.session_state.scraping = True
        st.session_state.interrumpido = False
        st.session_state.file_counter = 1  # Reiniciar numeración
        st.rerun()

# Proceso de scraping en curso
if st.session_state.scraping and not st.session_state.interrumpido:
    i = st.session_state.current_index
    urls = st.session_state.urls

    if i < len(urls):
        current_url = urls[i]
        st.markdown(f"### Procesando URL {i+1} de {len(urls)}")
        st.code(current_url)

        driver = init_driver()
        texto = extract_text_from_url(driver, current_url)
        driver.quit()

        if texto:
            # Eliminar números que NO estén entre corchetes
            texto = re.sub(r'(?<!\[)\b\d+\b(?!\])', '', texto)

            # Eliminar frases fijas no deseadas
            texto = texto.replace("The Latin Library", "")
            texto = texto.replace("The Classics Page", "")

            # Eliminar múltiples espacios que puedan quedar tras la limpieza
            texto = re.sub(r'\s{2,}', ' ', texto)

            nombre_fichero = f"{st.session_state.autor}_{st.session_state.obra}_{st.session_state.file_counter}.txt"
            path = os.path.join(st.session_state.output_folder, nombre_fichero)
            with open(path, "w", encoding="utf-8") as f:
                f.write(texto)
            st.success(f"Guardado: {nombre_fichero}")
            st.session_state.current_index += 1
            st.session_state.file_counter += 1
            time.sleep(1)
            st.rerun()

        else:
            st.error("❌ No se pudo scrapear la URL. Corrígela o interrumpe el proceso.")

            with st.form(f"form_fallo_{i}"):
                nueva_url = st.text_input("Corrige la URL fallida:", value=current_url, key=f"fix_url_{i}")
                nuevo_num = st.number_input("Nuevo número de fichero:", min_value=1, value=st.session_state.file_counter, step=1, key=f"fix_num_{i}")
                continuar = st.form_submit_button("✅ Continuar con URL corregida y nueva numeración")

            if continuar:
                st.session_state.urls[i] = nueva_url
                st.session_state.file_counter = nuevo_num
                st.rerun()

            if st.button("🛑 Interrumpir scraping"):
                st.session_state.interrumpido = True
                st.warning("Scraping interrumpido por el usuario.")

    else:
        # Todo completado, generar zip
        zip_path = os.path.join(st.session_state.output_folder, "Textos.zip")
        # Eliminar el ZIP si ya existe para evitar duplicados
        if os.path.exists(zip_path):
            os.remove(zip_path)

        with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            for filename in os.listdir(st.session_state.output_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(st.session_state.output_folder, filename)
                    zipf.write(file_path, arcname=filename)
        st.success("🎉 Scraping finalizado y archivos comprimidos en Textos.zip")
        with open(zip_path, "rb") as f:
            st.download_button("Descargar ZIP", f, file_name="Textos.zip")
        st.session_state.scraping = False
        st.session_state.current_index = 0
        st.session_state.file_counter = 1
