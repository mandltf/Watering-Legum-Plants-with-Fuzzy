import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import streamlit as st

data = pd.read_csv('dataset_tanaman.csv')

st.set_page_config(page_title='Projek Akhir')

col1, col2 = st.columns([1, 3.5])  # Adjust the ratio as needed

with col1:
    st.image('tanaman.png', width=160)

with col2:
    st.title('Sistem Prediksi Kebutuhan Penyiraman Legum')


if st.checkbox("Tampilkan Data Asli"):
    required_columns = ['label', 'unsur_hara', 'temperature', 'humidity', 'rainfall', 'ph']
    st.dataframe(data[required_columns].head(10))
st.markdown("---")

with st.expander('Grafik analitik'):
    st.subheader("Scatter Plot dan Box Plot")
    col1, col2, col3 = st.columns(3) 
    with col1:
        # 1. Scatter Plot
        fig3, ax3 = plt.subplots()
        labels = data['label'].unique()
        colors = plt.cm.get_cmap('tab10', len(labels))
        for i, label in enumerate(labels):
            subset = data[data['label'] == label]
            ax3.scatter(subset['temperature'], subset['humidity'], label=label, color=colors(i))
        ax3.set_xlabel("suhu")
        ax3.set_ylabel("kelembapan")
        ax3.set_title("Suhu vs Kelembaban")
        ax3.legend()
        st.pyplot(fig3)
        st.markdown('- **Mothbeans** : suhu 24–34°C dan kelembaban 40–70% dengan sebaran cukup luas')
        st.markdown('- **Mungbean** : terkonsentrasi pada kelembaban tinggi (85–90%), suhu 26–30°C')
        st.markdown('- **Blackgram** : terdistribusi pada suhu 25–34°C dan kelembaban 60–70%, lebih stabil daripada mothbeans')


    with col2:
        fig3, ax3 = plt.subplots()
        labels = data['label'].unique()
        colors = plt.cm.get_cmap('tab10', len(labels))
        for i, label in enumerate(labels):
            subset = data[data['label'] == label]
            ax3.scatter(subset['unsur_hara'], subset['ph'], label=label, color=colors(i))
        ax3.set_xlabel("unsur hara")
        ax3.set_ylabel("ph")
        ax3.set_title("Unsur hara vs pH")
        ax3.legend()
        st.pyplot(fig3)
        st.markdown('- **Mothbeans** : pH bervariasi (sekitar 5–10) dan unsur hara 20–50, tersebar luas')
        st.markdown('- **Mungbean** : pH stabil di kisaran 6.3–6.8 dengan unsur hara 25–35, cenderung terkonsentrasi')
        st.markdown('- **Blackgram** : unsur hara 35–50 dengan pH sekitar 6.8–7.5, relatif stabil')

    with col3:
        grouped = data.groupby('label')
        fig2, ax2 = plt.subplots()
        data_box = [group['rainfall'].values for name, group in grouped]
        ax2.boxplot(data_box, labels=grouped.groups.keys())
        ax2.set_title("Distribusi Curah Hujan per Jenis Tanaman")
        ax2.set_ylabel("Curah Hujan")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        st.markdown('- **Mothbeans** cocok di daerah dengan curah hujan tinggi dan stabil')
        st.markdown('- **Mungbean** lebih fleksibel (toleran berbagai curah hujan)')
        st.markdown('- **Blackgram** cocok di daerah dengan curah hujan sedang')


st.subheader("Dataset tanaman dengan 5 kriteria:")
col1, col2 = st.columns(2)  # Adjust the ratio as needed

with col1:
    st.write("- unsur hara (kurang, cukup, berlebih)")
    st.write("- temperature (rendah, sedang, tinggi)")
    st.write("- humidity (kering, normal, lembab)")

with col2:
    st.write("- rainfall (rendah, sedang, tinggi)")
    st.write("- ph (asam, netral, basa)")
st.markdown("---")

with st.sidebar:
    st.title("Profile 👤")
    col1, col2 = st.columns(2)  # Adjust the ratio as needed

    with col1:
        st.write('Nabila Putri S')
        st.write('Amanda Latifa')

    with col2:
        st.write(': 123230002')
        st.write(': 123230138')

    st.markdown("---")



unsur_hara = ctrl.Antecedent(np.arange(18,54,1), 'unsur') #input
temperature = ctrl.Antecedent(np.arange(24, 36, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(40, 91, 1), 'humidity') 
rainfall = ctrl.Antecedent(np.arange(30, 76, 1), 'rainfall') 
ph = ctrl.Antecedent(np.arange(3, 11, 1), 'ph')
kebutuhan_air = ctrl.Consequent(np.arange(0, 31, 1), 'kebutuhan_air')


unsur_hara['kurang'] = fuzz.trapmf(unsur_hara.universe, [18, 18, 25, 32])
unsur_hara['cukup'] = fuzz.trimf(unsur_hara.universe, [28, 35, 42])
unsur_hara['berlebih'] = fuzz.trapmf(unsur_hara.universe, [40, 45, 53, 53])

temperature['rendah'] = fuzz.trapmf(temperature.universe, [24, 24, 26, 28])
temperature['sedang'] = fuzz.trimf(temperature.universe, [26, 29.5, 33])
temperature['tinggi'] = fuzz.trapmf(temperature.universe, [31, 33, 35, 35])

humidity['kering'] = fuzz.trapmf(humidity.universe, [40, 40, 50, 60])
humidity['normal'] = fuzz.trimf(humidity.universe, [55, 65, 75])
humidity['lembab'] = fuzz.trapmf(humidity.universe, [70, 80, 90, 90])

rainfall['rendah'] = fuzz.trapmf(rainfall.universe, [30, 30, 40, 50])
rainfall['sedang'] = fuzz.trimf(rainfall.universe, [45, 58, 70])
rainfall['tinggi'] = fuzz.trapmf(rainfall.universe, [65, 70, 75, 75])

ph['asam'] = fuzz.trapmf(ph.universe, [3, 3, 4, 5])
ph['netral'] = fuzz.trimf(ph.universe, [5, 6, 8])
ph['basa'] = fuzz.trapmf(ph.universe, [7, 8, 10, 10])

kebutuhan_air['sedikit'] = fuzz.trimf(kebutuhan_air.universe, [0, 0, 10])
kebutuhan_air['sedang']  = fuzz.trimf(kebutuhan_air.universe, [5, 15, 25])
kebutuhan_air['banyak']  = fuzz.trimf(kebutuhan_air.universe, [20, 30, 30])

rules = [
    # Aturan 1–50
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedang']),

    # Aturan 51–100
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedang']),

    # Aturan 101–150
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['cukup'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['rendah'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),

    # Aturan 151–200
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['kurang'] & temperature['sedang'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),

    # Aturan 201–243
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['kering'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['banyak']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedang']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['normal'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['rendah'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['sedang'] & ph['basa'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['asam'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['netral'], kebutuhan_air['sedikit']),
    ctrl.Rule(unsur_hara['berlebih'] & temperature['tinggi'] & humidity['lembab'] & rainfall['tinggi'] & ph['basa'], kebutuhan_air['sedikit']),

]

# Sistem kontrol fuzzy
watering_legum = ctrl.ControlSystem(rules)
watering = ctrl.ControlSystemSimulation(watering_legum)


# selectedRow = st.slider('Pilih row', 0, 300, 150)
# row = data.iloc[selectedRow]
with st.sidebar:
    row_options = [f"{i} - {data.iloc[i]['label']}" for i in data.index]
    selected_row = st.selectbox("Pilih Baris Data", row_options)

# Ambil index
row_index = int(selected_row.split(" - ")[0])
row = data.iloc[row_index]

watering.input['unsur'] = row['unsur_hara']
watering.input['temperature'] = row['temperature']
watering.input['humidity'] = row['humidity']
watering.input['rainfall'] = row['rainfall']
watering.input['ph'] = row['ph']
# Hitung fuzzy
watering.compute()
hasil = watering.output['kebutuhan_air']

st.subheader("Hasil Prediksi Kebutuhan Air")
col1, col2, col3, col4 = st.columns(4)  # Adjust the ratio as needed

with col1:
    st.write(f"- suhu")
    st.write(f"- kelembapan")
    st.write(f"- curah hujan")

with col2:
    st.write(f": **{row['temperature']:.2f}°C**")
    st.write(f": **{row['humidity']:.2f}%**")
    st.write(f": **{row['rainfall']:.2f} mm**")

with col3:
    st.write(f"- pH tanah")
    st.write(f"- unsur hara")

with col4:
    st.write(f": **{row['ph']:.2f}**")
    st.write(f": **{row['unsur_hara']:.2f}**")


def plot_and_display_output(variable, title, result):
    fig, ax = plt.subplots()
    for term_name in variable.terms:
        term_mf = variable[term_name].mf
        ax.plot(variable.universe, term_mf, label=term_name)
    ax.grid(True, alpha=0.35)
    ax.set_title(title, loc='center', fontsize=14)
    ax.set_xlabel(variable.label)
    ax.set_ylabel('Derajat Keanggotaan')
    # Menambahkan garis tengah hasil
    ax.axvline(x = result, color='black', linestyle='--', label=f'Hasil : {result:.2f}')
    ax.legend()
    st.pyplot(fig)


rekomen_label = max(kebutuhan_air.terms, key=lambda term: fuzz.interp_membership(
    kebutuhan_air.universe, kebutuhan_air[term].mf, hasil))  # hasil = output numeric

if st.sidebar.button("Grafik Keanggotaan Perhitungan Fuzzy"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Unsur Hara','Temperature','Humidity', 'Rainfall', 'pH'])
    with tab1:
        plot_and_display_output(unsur_hara, "Unsur Hara", row['unsur_hara'])
    with tab2:
        plot_and_display_output(temperature, "Temperature", row['temperature'])
    with tab3:
        plot_and_display_output(humidity, "Humidity", row['humidity'])
    with tab4:
        plot_and_display_output(rainfall, "Rainfall", row['rainfall'])
    with tab5:
        plot_and_display_output(ph, "pH", row['ph'])
    st.sidebar.success(f'Maka kebutuhan air tanaman adalah **{watering.output['kebutuhan_air']:.2f} mm/m³** atau dapat dikategorikan ***{rekomen_label}***')
