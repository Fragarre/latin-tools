import streamlit as st 
st.set_page_config(layout="wide", page_title="Latin Tools")
pages = {
    "Herramientas": [    
        st.Page("scrapper.py", title="Extraer textos"),
        st.Page("translate.py", title="Traducción"),
        st.Page("stylometrics.py", title="Estilometría"),
        st.Page("author.py", title="Autoría"),
    ]
}

pg = st.navigation(pages)
pg.run()


