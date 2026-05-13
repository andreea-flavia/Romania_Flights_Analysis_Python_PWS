import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Proiect Analiza Zboruri",
    layout="centered",
    initial_sidebar_state="expanded")
st.title("Analiza Preturilor Avioanelor - Romania")
st.subheader("1 Mai - 1 Iunie 2026")
st.markdown("Acesta aplicatie analizeaza factorii economici care influenteaza tarifele aeriene.")
st.divider();

@st.cache_data
def load_data():
    return pd.read_csv('flights_ro_details.csv')


def pagina_prelucrare():
    df_ro = load_data()
    st.header("1. Incarcarea setului de date")
    if st.checkbox("Arata setul de date brut"):
        st.markdown("Coloana 'scraped_by' reprezinta data in care au fost preluate datele din set - 01.05.2026")
        st.write(df_ro.head(10))

    #----------------------------------------------------------------------------------------------
    st.divider()
    st.header("2. Tratarea valorilor lipsa si extreme")

    st.code('''
    # Eliminarea randurilor cu date lipsa pe coloanele cheie
    df_ro = df_ro.dropna(subset=['price', 'days_left', 'duration_minutes'])
    
    # Eliminarea valorilor extreme (Outliers) folosind percentila 99
    limita_sup = df_ro['price'].quantile(0.99)
    df_ro = df_ro[df_ro['price'] < limita_sup]
        ''', language='python')

    initial_count = len(df_ro)
    df_ro = df_ro.dropna(subset=['price', 'days_left', 'duration_minutes'])
    nan_count = initial_count - len(df_ro)
    limita_sup = df_ro['price'].quantile(0.99)
    df_final = df_ro[df_ro['price'] < limita_sup].copy()
    outlier_count = len(df_ro) - len(df_final)
    df_ro = df_final

    c1, c2, c3 = st.columns(3)
    c1.metric("Total inregistrari", initial_count)
    c2.metric("Valori lipsa eliminate", nan_count)
    c3.metric("Outliers eliminati (zboruri cu tarife extreme trecute de prag)", outlier_count)
    st.info(f"Pragul stabilit pentru preturi extreme: {limita_sup:.2f} EUR -> outliers")

    #----------------------------------------------------------------------------------------------
    st.divider()
    st.header("3. Codificarea datelor (Encoding)")

    st.code('''
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # Transformam variabilele text in numere
    df_ro['airline_enc'] = le.fit_transform(df_ro['airline'])
    df_ro['class_enc'] = le.fit_transform(df_ro['class']) #exista doar clasa 'economy'
    df_ro['source_enc'] = le.fit_transform(df_ro['source_city'])
        ''', language='python')

    le = LabelEncoder()
    df_ro['airline_enc'] = le.fit_transform(df_ro['airline'])
    df_ro['class_enc'] = le.fit_transform(df_ro['class'])
    df_ro['source_enc'] = le.fit_transform(df_ro['source_city'])

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dictionar Linii Aeriene (Top 5):**")
        st.dataframe(df_ro[['airline', 'airline_enc']].drop_duplicates().head(5))

    with col2:
        st.write("**Dictionar Orase Plecare (Top 5):**")
        st.dataframe(df_ro[['source_city', 'source_enc']].drop_duplicates().sort_values('source_enc').head(5))

    #----------------------------------------------------------------------------------------------
    st.divider()
    st.header("4. Metode de scalare")
    st.code('''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Scalam durata zborului si numarul de zile ramase
    df_ro['duration_scaled'] = scaler.fit_transform(df_ro[['duration_minutes']])
    df_ro['days_left_scaled'] = scaler.fit_transform(df_ro[['days_left']])
        ''', language='python')

    scaler = StandardScaler()
    df_ro['duration_scaled'] = scaler.fit_transform(df_ro[['duration_minutes']])
    df_ro['days_left_scaled'] = scaler.fit_transform(df_ro[['days_left']])

    st.write("**Comparatie date originale vs. date scalate (Top 5):**")
    st.dataframe(df_ro[['airline','route','departure_time','duration_minutes', 'duration_scaled', 'days_left', 'days_left_scaled']].head())

    st.session_state['df_curat'] = df_ro

def pagina_statistici():
    st.title("Statistici si Agregari")
    if 'df_curat' in st.session_state:
        df_ro = st.session_state['df_curat']
    else:
        df_ro = load_data()

    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("5. Analiza per companie aeriana")
    airline_stats = df_ro.groupby('airline').agg(
        zboruri=('price', 'size'),
        pret_mediu=('price', 'mean'),
        pret_median=('price', 'median')
    ).reset_index().sort_values('pret_mediu', ascending=False)




    st.write("**Tabel comparativ tarife:**")
    st.dataframe(airline_stats)


    fig, ax = plt.subplots()
    sns.barplot(data=airline_stats.head(10), y='airline', x='pret_mediu', palette='rocket', ax=ax)
    ax.set_title("Top 10 Companii dupa pretul mediu (EUR)")
    st.pyplot(fig)

    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("6. Influenta numarului de escale asupra tarifului")

    # Calculam media pentru fiecare grup de escale
    stops_avg = df_ro.groupby('stops')['price'].mean().reset_index()

    # Centrarea graficului folosind coloane
    c1, c2, c3 = st.columns([1, 3, 1])

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=stops_avg, x='stops', y='price', palette='magma', ax=ax)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f} €',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points')
        ax.set_title("Pretul mediu in functie de numarul de escale")
        ax.set_ylabel("Pret mediu (EUR)")
        ax.set_xlabel("Numar de escale")
        st.pyplot(fig)

    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("7. Evolutia pretului in functie de momentul rezervarii")
    lead_curve = df_ro.groupby('days_left')['price'].mean().reset_index()

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=lead_curve, x='days_left', y='price', color='blue', linewidth=2, ax=ax3)
    ax3.set_title("Pretul biletului creste pe masura ce ne apropiem de data zborului")
    ax3.set_xlabel("Zile ramase pana la decolare")
    ax3.set_ylabel("Pret mediu (EUR)")
    st.pyplot(fig3)

    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("8. Analiza pe momentele zilei")

    daypart_order = ['morning', 'afternoon', 'evening', 'night']
    daypart_stats = df_ro.groupby('departure_daypart')['price'].median().reindex(daypart_order).reset_index()

    fig4, ax4 = plt.subplots()
    sns.barplot(data=daypart_stats, x='departure_daypart', y='price', palette='crest', ax=ax4)
    ax4.set_title("Pretul median in functie de momentul plecarii")
    st.pyplot(fig4)



def pagina_modele():
    st.title("Modele ML si Predictii")
    if 'df_curat' in st.session_state:
        df_ro = st.session_state['df_curat']
    else:
        df_ro = load_data()
    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("7. Segmentarea Pietei (Clusterizare K-Means)")
    X_km = df_ro[['duration_minutes', 'price']]
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_ro['cluster'] = km.fit_predict(X_km)

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(df_ro['duration_minutes'], df_ro['price'],
                             c=df_ro['cluster'], cmap='viridis', alpha=0.6)
        ax.set_xlabel("Durata (minute)")
        ax.set_ylabel("Pret (EUR)")
        ax.set_title("Segmentarea zborurilor: Economy vs Premium vs Long-haul")
        plt.colorbar(scatter, label='Cluster ID')
        st.pyplot(fig)

    st.success(
        "**Interpretare Clusterizare:** Clusterul cu pret mic si durata scurta reprezinta zborurile regionale (low-cost), in timp ce clusterele cu pret ridicat identifica zborurile business sau de lunga durata.")

    # ----------------------------------------------------------------------------------------------
    st.divider()
    st.header("8. Analiza de Regresie Multipla (Statsmodels)")

    features = ['days_left', 'stops', 'duration_minutes', 'airline_enc']
    X = df_ro[features]
    X = sm.add_constant(X)
    y = df_ro['price']

    model = sm.OLS(y, X).fit()
    st.subheader("Rezumat Model OLS (Ordinary Least Squares)")
    st.text(str(model.summary()))


# --- CONFIGURARE NAVIGARE ---

pg = st.navigation([
    st.Page(pagina_prelucrare, title="I. Prelucrare Date"),
    st.Page(pagina_statistici, title="II. Statistici & Grafice"),
    st.Page(pagina_modele, title="III. Modele Predictive"),
])

pg.run()


