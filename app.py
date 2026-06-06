import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import re

# ==========================================
# 1. CONFIGURACIÓN Y EMPATÍA (Recomendación Clase 9)
# ==========================================
st.set_page_config(
    page_title="Dashboard - ML & Football Scouting",
    page_icon="⚽",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size:18px !important; color: #4B5563; text-align: center; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# Panel Lateral: Explicación para el usuario (Evitando la pantalla vacía)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5323/5323871.png", width=100)
    st.header("🔍 Contexto de Investigación")
    
    st.markdown("**¿De qué trata este Dashboard?**")
    st.info("Explora cómo los algoritmos supervisados de Machine Learning optimizan la precisión del scouting y la identificación de talento emergente en el fútbol profesional.")
    
    st.markdown("**Keywords Scopus:**")
    st.code('"Machine learning"\n"Scouting"\n"Talent identification"\n"Soccer"')
    
    st.markdown("---")
    st.subheader("📥 Carga tu propio Dataset")
    st.markdown("Sube tu archivo `.csv` exportado de Scopus con las columnas: Authors, Title, Year, Abstract, Cited by.")
    uploaded_file = st.file_uploader("", type=["csv"])

# --- CARGA OPTIMIZADA DE DATOS ---
@st.cache_data
def load_data(file_source):
    df = pd.read_csv(file_source)
    return df

# ==========================================
# 2. CUERPO PRINCIPAL DEL DASHBOARD
# ==========================================
st.markdown('<div class="main-title">⚽ Machine Learning en Scouting de Fútbol</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis del Estado del Arte de la Literatura Científica</div>', unsafe_allow_html=True)

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("✅ Dataset cargado correctamente.")
else:
    # Opción automática para que el usuario no vea una pantalla rota si no sube nada
    try:
        # Aquí puedes colocar la ruta raw de tu GitHub
        github_url = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/scopus_export.csv"
        df = pd.read_csv(github_url)
        st.info("📊 Mostrando datos de ejemplo desde GitHub. Sube tu archivo CSV en la barra lateral para analizar nuevos datos.")
    except:
        st.warning("⚠️ Sube tu archivo CSV de Scopus desde el panel lateral izquierdo para activar los gráficos y el análisis.")

if df is not None:
    # ==========================================
    # 3. MÉTRICAS Y GRÁFICOS (Limpieza sugerida en Clase 9)
    # ==========================================
    
    # Resumen Rápido
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Publicaciones", len(df))
    col2.metric("Años de Análisis", f"{df['Year'].min()} - {df['Year'].max()}")
    col3.metric("Total Citaciones Acumuladas", int(df['Cited by'].sum()))

    # Pestañas para orden
    tab1, tab2, tab3 = st.tabs(["📋 Explorador de Datos", "📈 Tendencias y Citas", "🔤 Análisis de Abstracts"])

    with tab1:
        st.subheader("Filtro de Artículos Científicos")
        st.markdown("Se han omitido columnas con datos faltantes para mostrarte solo la metadata esencial.")
        
        # Filtro de año
        selected_year = st.slider("Filtrar publicaciones desde el año:", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=int(df['Year'].min()))
        df_filtered = df[df['Year'] >= selected_year]
        
        # Mostrar columnas limpias y relevantes
        columns_to_show = ['Authors', 'Title', 'Year', 'Source title', 'Cited by', 'DOI']
        st.dataframe(df_filtered[columns_to_show], use_container_width=True)

    with tab2:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Publicaciones por Año")
            year_counts = df['Year'].value_counts().sort_index()
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax1, palette="Blues_d")
            ax1.set_ylabel("Cantidad")
            ax1.set_xlabel("Año")
            st.pyplot(fig1)

        with col_g2:
            st.subheader("Top 5 Artículos más Citados")
            # Ordenar para gráfica
            top_cited = df[['Title', 'Cited by']].sort_values(by='Cited by', ascending=False).head(5)
            top_cited = top_cited.sort_values(by='Cited by', ascending=True)
            top_cited['Short Title'] = top_cited['Title'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Cited by', y='Short Title', data=top_cited, ax=ax2, palette="viridis")
            ax2.set_xlabel("Citaciones")
            ax2.set_ylabel("")
            st.pyplot(fig2)

    with tab3:
        st.subheader("Análisis de Frecuencia en Resúmenes (Abstracts)")
        st.markdown("Identifica las metodologías más mencionadas en las investigaciones extraídas.")
        
        # Script de Limpieza de texto (Recomendación del profe)
        words_list = []
        stopwords = set(['the', 'a', 'in', 'of', 'and', 'to', 'is', 'for', 'with', 'on', 'by', 'at', 'an', 'this', 'that', 'from', 'as', 'are', 'it', 'we', 'player', 'players', 'football', 'soccer', 'scouting', 'talent', 'identification', 'machine', 'learning', 'data', 'using', 'used', 'analysis', 'team', 'sports', 'study', 'results', 'proposed', 'performance', 'our', 'based', 'was', 'were', 'their', 'how', 'which'])
        
        for abstract in df['Abstract'].dropna():
            words = re.findall(r'\b\w+\b', abstract.lower())
            for w in words:
                if w not in stopwords and len(w) > 2:
                    words_list.append(w)
                    
        word_counts = collections.Counter(words_list).most_common(10)
        
        if word_counts:
            words_df = pd.DataFrame(word_counts, columns=['Palabra', 'Frecuencia']).sort_values(by='Frecuencia', ascending=True)
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Frecuencia', y='Palabra', data=words_df, palette="rocket", ax=ax3)
            ax3.set_xlabel("Frecuencia")
            ax3.set_ylabel("Palabra Clave")
            st.pyplot(fig3)
        else:
            st.info("No hay suficientes datos en los resúmenes para generar el gráfico.")