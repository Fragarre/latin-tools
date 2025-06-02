# 📚 Herramientas de análisis de Textos en Latín

## 📖 Referencias y créditos

Este proyecto se inspira en el trabajo de Benjamin Nagy sobre estilometría aplicada a textos latinos. Agradezco especialmente su artículo:

**Nagy, Benjamin.**  
*Some stylometric remarks on Ovid’s Heroides and the Epistula Sapphus*.  
Digital Scholarship in the Humanities, 2023.  
[https://doi.org/10.1093/llc/fqac098](https://doi.org/10.1093/llc/fqac098)

El repositorio asociado al artículo original está licenciado bajo CC-BY 4.0, lo que permite su reutilización con atribución adecuada. Parte del enfoque metodológico y elementos del código fueron adaptados de dicho trabajo con respeto a esta licencia:  
[https://github.com/bnagy/heroides-paper](https://github.com/bnagy/heroides-paper)

---
### 📚 1. Extracción automatizada de textos clásicos
Extrae de forma estructurada capítulos completos de obras clásicas desde páginas como [The Latin Library](https://www.thelatinlibrary.com/), utilizando un patrón de URLs secuenciales.

### 🔄 2. Traducción Latín ↔ Español
Traduce fragmentos de texto o archivos completos entre **latín clásico** y **español moderno**.

- Compatible con archivos `.txt` de hasta 50.000 caracteres.

### 🧠 3. Análisis estilométrico
Análisis de un conjunto de textos latinos en formato .txt, comprimidos en un archivo .zip

- **Análisis de similitud y clustering**: t-SNE, UMAP

### 🧠 4. Análisis de Autoría con probabilidades por autores.
- Entrenamiento de modelo con textos etiquetados por autor.
- Evaluación del modelo con conjunto de test dividido automáticamente.
- Matriz de confusión para ver el rendimiento por clase.
- Detección y listado de errores de clasificación.
- Clasificación de nuevos textos y visualización de:
  - Distancias a centroides.
  - Probabilidades de autoría.
  - Gráficos de barras interactivos por texto.