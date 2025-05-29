import streamlit as st
import toml
import textwrap
import tiktoken
from openai import OpenAI
import math

# --- Cargar API Key ---
try:
    config = toml.load("config.toml")
    api_key = config["openai"]["api_key"]
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ No se pudo cargar la API key: {e}")
    st.stop()

# --- Configuración inicial ---
MAX_CHARS_ARCHIVO = 50000
COSTES_MODELOS = {
    "gpt-4o": 0.005,   # $/1K tokens (input)
    "gpt-4.1-mini": 0.03,     # $/1K tokens (input)
}

# --- Interfaz UI ---
st.subheader("Traductor Latín ↔ Español")

st.markdown(
    "Traduce textos clásicos entre **latín** y **español**.\n\n"
    "- Texto Manual: Máximo por texto  : 1.000 caracteres\n"
    "- Archivo .txt: Máximo por archivo: 50.000 caracteres"
)

modo = st.selectbox("Modo de traducción:", ["Latín → Español", "Español → Latín"])
# modelo = st.radio("Modelo:", ["GPT-4.1 mini (gpt-4o)", "GPT-4.0"], horizontal=True)
modelo_seleccionado = "gpt-4o" # if "mini" in modelo else "gpt-4"
precio_token = COSTES_MODELOS[modelo_seleccionado]

input_tipo = st.radio("Entrada:", ["Texto manual", "Archivo .txt"], horizontal=True)

texto_entrada = ""

if input_tipo == "Texto manual":
    placeholder = (
        "Ejemplo: Gallia est omnis divisa in partes tres..."
        if "Latín" in modo
        else "Ejemplo: Toda la Galia está dividida en tres partes..."
    )
    texto_entrada = st.text_area("Texto a traducir: ", placeholder=placeholder, height=200, max_chars=3000)
    st.markdown("**Después de introducir el texto, pulsa 'control + enter'**")
else:
    archivo = st.file_uploader("Sube un archivo .txt (máx. 30.000 caracteres)", type=["txt"])
    if archivo:
        contenido = archivo.read().decode("utf-8")
        total_caracteres = len(contenido)
        if total_caracteres > MAX_CHARS_ARCHIVO:
            st.warning(f"⚠️ El archivo supera los {MAX_CHARS_ARCHIVO} caracteres. Solo se traducirán los primeros {MAX_CHARS_ARCHIVO} de los {total_caracteres} totales.")
            texto_entrada = contenido[:MAX_CHARS_ARCHIVO]
        else:
            texto_entrada = contenido
        st.success(f"✅ Archivo cargado. Total de caracteres a traducir: {len(texto_entrada)}")

# --- Estimación previa y botón de confirmación ---
if texto_entrada.strip():

    bloques = textwrap.wrap(texto_entrada, width=1000, break_long_words=False, break_on_hyphens=False)
    encoding = tiktoken.encoding_for_model(modelo_seleccionado)
    total_tokens_estimado = 0

    if modo == "Latín → Español":
        system_prompt = (
            "Eres un traductor experto en latín clásico. Traduce los textos latinos al español moderno "
            "con fidelidad, claridad y estilo natural."
        )
        encabezado = "Traduce al español el siguiente texto en latín:\n\n"
    else:
        system_prompt = (
            "Eres un traductor experto en latín clásico. Traduce textos españoles al latín clásico "
            "con corrección gramatical y estilo adecuado a los autores clásicos."
        )
        encabezado = "Traduce al latín clásico el siguiente texto en español:\n\n"

    for bloque in bloques:
        prompt = encabezado + bloque
        total_tokens_estimado += len(encoding.encode(prompt)) + len(encoding.encode(system_prompt))

    coste_estimado = (total_tokens_estimado / 1000) * precio_token
    tiempo_estimado_min = math.ceil((len(texto_entrada) / 10000) * 1.2)

    st.markdown(f"💸 **Coste estimado:** ${coste_estimado:.4f} USD")
    st.markdown(f"⏳ **Tiempo estimado de traducción:** {tiempo_estimado_min} minutos")
    
    if st.button("✅ Confirmar y traducir"):
        with st.spinner("🔄 Traduciendo..."):
            traducciones = []
            try:
                for i, bloque in enumerate(bloques, 1):
                    st.info(f"🧩 Traduciendo bloque {i} de {len(bloques)}...")

                    prompt = encabezado + bloque
                    response = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1000,
                    )
                    traduccion = response.choices[0].message.content.strip()
                    traducciones.append(traduccion)

                resultado = "\n\n".join(traducciones)
                st.success("✅ Traducción completada.")

                st.text_area("Resultado:", resultado, height=400)
                st.download_button(
                    "📥 Descargar traducción",
                    data=resultado,
                    file_name="traduccion.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"❌ Error al traducir: {e}")
else:
    st.info("Introduce texto o carga un archivo para obtener la estimación de tiempo y coste.")

