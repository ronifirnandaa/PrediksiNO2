import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta

# ===============================
# Load Model, Scaler, dan Dataset
# ===============================
knn_h3 = joblib.load("model_h3.pkl")
scaler_h3 = joblib.load("scaler_h3.pkl")
no2 = pd.read_csv("data_no2.csv")
no2['t'] = pd.to_datetime(no2['t'])
no2 = no2.sort_values('t')
X = no2[['day_index']]
y_scaled = no2['NO2_scaled']

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Prediksi NO₂ KNN", layout="wide")
st.title("Prediksi Konsentrasi NO₂ (KNN)")

tab1, tab2 = st.tabs(["Prediksi Otomatis", "Prediksi Manual"])

# -------------------------------
# Tab 1: Prediksi Otomatis
# -------------------------------
with tab1:
    st.subheader("Prediksi Hari Berikutnya")
    
    # Prediksi hari berikutnya
    next_day = no2['t'].iloc[-1] + timedelta(days=1)
    next_input = np.array([[no2['day_index'].iloc[-1] + 1]])
    next_pred_scaled = knn_h3.predict(next_input)[0]
    next_pred = scaler_h3.inverse_transform([[next_pred_scaled]])[0][0]
    
    # Tentukan kategori
    median_val = no2['NO2'].quantile(0.50)
    upper_quantile_val = no2['NO2'].quantile(0.75)
    
    if next_pred <= median_val:
        kategori = "Baik"
        penjelasan = f"Nilai ini berada di bawah atau sama dengan batas baik (median: {median_val:.6f})."
    elif next_pred <= upper_quantile_val:
        kategori = "Sedang"
        penjelasan = f"Nilai ini di atas median, tapi di bawah kuantil atas ({upper_quantile_val:.6f})."
    else:
        kategori = "Tinggi (Tidak Baik)"
        penjelasan = f"Nilai ini berada di atas 75% dari data historis (lebih tinggi dari {upper_quantile_val:.6f})."
    
    # Tampilkan hasil otomatis
    st.metric("Tanggal Prediksi", next_day.strftime("%d-%m-%Y"))
    st.metric("Prediksi NO₂ (µg/m³)", f"{next_pred:.10f}")
    st.write(f"**Kategori Prediksi:** {kategori}")
    st.write(f"**Penjelasan:** {penjelasan}")
    
# -------------------------------
# Tab 2: Prediksi Manual
# -------------------------------
manual_pred = None
manual_day = None

with tab2:
    st.subheader("Prediksi Manual")
    st.write("Masukkan indeks hari atau tanggal untuk memprediksi NO₂")
    
    # Input user
    use_index = st.checkbox("Gunakan indeks hari (day_index)", value=True)
    
    if use_index:
        day_index_input = st.number_input("Masukkan day_index:", min_value=0, value=int(no2['day_index'].iloc[-1]+1))
        input_array = np.array([[day_index_input]])
        manual_day = no2['t'].iloc[0] + pd.to_timedelta(day_index_input, unit='D')
    else:
        date_input = st.date_input("Masukkan tanggal:")
        # Hitung day_index dari data awal
        day_index_input = (pd.to_datetime(date_input) - no2['t'].iloc[0]).days
        input_array = np.array([[day_index_input]])
        manual_day = pd.to_datetime(date_input)
    
    if st.button("Prediksi Manual"):
        manual_pred_scaled = knn_h3.predict(input_array)[0]
        manual_pred = scaler_h3.inverse_transform([[manual_pred_scaled]])[0][0]
        
        # Kategori
        if manual_pred <= median_val:
            kategori_manual = "Baik"
            penjelasan_manual = f"Nilai ini berada di bawah atau sama dengan batas baik (median: {median_val:.6f})."
        elif manual_pred <= upper_quantile_val:
            kategori_manual = "Sedang"
            penjelasan_manual = f"Nilai ini di atas median, tapi di bawah kuantil atas ({upper_quantile_val:.6f})."
        else:
            kategori_manual = "Tinggi (Tidak Baik)"
            penjelasan_manual = f"Nilai ini berada di atas 75% dari data historis (lebih tinggi dari {upper_quantile_val:.6f})."
        
        st.metric("Tanggal Manual", manual_day.strftime("%d-%m-%Y"))
        st.metric("Prediksi NO₂ (µg/m³)", f"{manual_pred:.10f}")
        st.write(f"**Kategori Prediksi:** {kategori_manual}")
        st.write(f"**Penjelasan:** {penjelasan_manual}")

# ===============================
# Plot Data & Prediksi Dinamis
# ===============================
st.subheader("Visualisasi Data & Prediksi")
fig, ax = plt.subplots(figsize=(12,6))

# Data asli
ax.plot(no2['t'], no2['NO2'], color='#7A7A7A', linestyle='-', linewidth=2.0, label='Data Asli')
ax.scatter(no2['t'], no2['NO2'], color='#FFA500', s=80, edgecolor='black', linewidth=0.6, label='Titik NO₂')

# Garis prediksi KNN seluruh data
y_pred_full = knn_h3.predict(X)
y_pred_full_inv = scaler_h3.inverse_transform(y_pred_full.reshape(-1,1))
ax.plot(no2['t'], y_pred_full_inv, color='#00CC66', linestyle='--', linewidth=2.0, label='Prediksi KNN')

# Titik prediksi otomatis
ax.scatter(next_day, next_pred, color='#007ACC', s=120, edgecolor='white', linewidth=1.2, zorder=5)
ax.text(next_day, next_pred, f"{next_pred:.8f}", color='#007ACC', fontsize=11, fontweight='bold', ha='left', va='bottom')

# Titik prediksi manual (jika ada)
if manual_pred is not None and manual_day is not None:
    ax.scatter(manual_day, manual_pred, color='#FF33AA', s=120, edgecolor='white', linewidth=1.2, zorder=5)
    ax.text(manual_day, manual_pred, f"{manual_pred:.8f}", color='#FF33AA', fontsize=11, fontweight='bold', ha='left', va='bottom')

ax.set_title("Prediksi Konsentrasi NO₂ (KNN)", fontsize=16, fontweight='bold')
ax.set_xlabel("Tanggal", fontsize=12)
ax.set_ylabel("Konsentrasi NO₂ (µg/m³)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=False)
st.pyplot(fig)
