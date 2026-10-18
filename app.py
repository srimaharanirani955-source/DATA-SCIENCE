
import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="USD/IDR | Economic Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# WARNA DAN STYLE
# =========================
BG = "#0B1120"
PANEL = "#111C31"
CYAN = "#38D9F9"
BLUE = "#5687FF"
TEXT = "#E8F0FF"
MUTED = "#8EA2C8"

st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #E8F0FF;
}
[data-testid="stSidebar"] {
    background: #101A2D;
    border-right: 1px solid #243653;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #E8F0FF !important;
}
p, label, span {
    color: #D4DFF5;
}
.hero {
    background: linear-gradient(120deg, #152B4A, #123C53);
    border: 1px solid #28556E;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 7px;
}
.hero-sub {
    color: #A9C5E7;
    font-size: 14px;
}
.kpi {
    background: #111C31;
    border: 1px solid #263A59;
    border-radius: 15px;
    padding: 20px;
    min-height: 132px;
}
.kpi-label {
    color: #9FB3D6;
    font-size: 13px;
    margin-bottom: 12px;
}
.kpi-value {
    color: #38D9F9;
    font-size: 25px;
    font-weight: 800;
    overflow-wrap: anywhere;
}
.kpi-note {
    color: #8197BD;
    font-size: 12px;
    margin-top: 8px;
}
.section-title {
    font-size: 21px;
    font-weight: 750;
    margin-top: 16px;
    margin-bottom: 10px;
    color: #E8F0FF;
}
div[data-testid="stPlotlyChart"] {
    background: #111C31;
    border: 1px solid #263A59;
    border-radius: 14px;
    padding: 8px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #263A59;
    border-radius: 12px;
}
div.stButton > button {
    border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# KONEKSI DATABASE
# =========================
@st.cache_data(ttl=60)
def load_data():
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3307,
        user="root",
        password="root",
        database="kurs_db",
        connection_timeout=10
    )

    try:
        query = """
            SELECT
                period,
                kurs_usdidr_rata_rata,
                inflasi_persen,
                cadangan_devisa_usd_juta
            FROM dataset_gabungan
            ORDER BY period
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    df["period"] = pd.to_datetime(
        df["period"], format="%Y-%m", errors="coerce"
    )
    for col in [
        "kurs_usdidr_rata_rata",
        "inflasi_persen",
        "cadangan_devisa_usd_juta"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["period"]).sort_values("period")
    return df


# =========================
# MEMUAT MODEL ML DAN FITUR
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_kurs.pkl"
FITUR_PATH = BASE_DIR / "fitur_model.pkl"

@st.cache_resource
def load_ml_assets():
    model = joblib.load(MODEL_PATH)
    fitur = joblib.load(FITUR_PATH)
    return model, fitur


# =========================
# STYLE GRAFIK
# =========================
def style_fig(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=25, r=25, t=55, b=25),
        title_font=dict(size=16, color=TEXT),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT)
        ),
        xaxis=dict(
            gridcolor="#253550",
            zerolinecolor="#253550"
        ),
        yaxis=dict(
            gridcolor="#253550",
            zerolinecolor="#253550"
        )
    )
    return fig


def kpi_card(label, value, note):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 📈 ECONOMIC INTELLIGENCE")
st.sidebar.caption("USD/IDR • Indonesia")
st.sidebar.divider()

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    data = load_data()
except Exception as e:
    st.error(
        "Dashboard belum bisa mengambil data dari MySQL. "
        "Pastikan container mysql-api-kurs aktif dan port 3307 tersedia."
    )
    st.code(str(e))
    st.stop()

if data.empty:
    st.warning("Tabel dataset_gabungan belum memiliki data.")
    st.stop()

tanggal_min = data["period"].min().date()
tanggal_max = data["period"].max().date()

st.sidebar.markdown("### Filter periode")
rentang = st.sidebar.date_input(
    "Pilih rentang bulan",
    value=(tanggal_min, tanggal_max),
    min_value=tanggal_min,
    max_value=tanggal_max
)

if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
    mulai, selesai = rentang
    filtered = data[
        (data["period"].dt.date >= mulai) &
        (data["period"].dt.date <= selesai)
    ].copy()
else:
    filtered = data.copy()

st.sidebar.divider()
st.sidebar.caption("Sumber data")
st.sidebar.markdown(
    "- Kurs: Frankfurter API\n"
    "- Inflasi: BPS\n"
    "- Cadangan devisa: FRED / IMF"
)
st.sidebar.caption("Data diperbarui melalui pipeline ETL.")


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">📊 USD/IDR Economic Intelligence</div>
    <div class="hero-sub">
        Dashboard analisis nilai tukar rupiah berdasarkan inflasi
        dan cadangan devisa Indonesia.
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Periode terpilih: {filtered['period'].min():%B %Y}"
    f" — {filtered['period'].max():%B %Y}"
    if not filtered.empty else "Tidak ada data pada periode ini."
)

if filtered.empty:
    st.warning("Tidak ada data pada rentang periode yang dipilih.")
    st.stop()

# =========================
# KPI
# =========================
kurs_valid = filtered["kurs_usdidr_rata_rata"].dropna()
inflasi_valid = filtered["inflasi_persen"].dropna()
devisa_valid = filtered["cadangan_devisa_usd_juta"].dropna()

a, b, c, d = st.columns(4)

with a:
    nilai = kurs_valid.iloc[-1] if not kurs_valid.empty else None
    kpi_card(
        "💱 Kurs USD/IDR terakhir",
        f"Rp {nilai:,.2f}" if nilai is not None else "N/A",
        "Nilai terakhir yang tersedia"
    )

with b:
    nilai = kurs_valid.mean() if not kurs_valid.empty else None
    kpi_card(
        "📉 Rata-rata kurs",
        f"Rp {nilai:,.2f}" if nilai is not None else "N/A",
        "Rata-rata periode terpilih"
    )

with c:
    nilai = inflasi_valid.iloc[-1] if not inflasi_valid.empty else None
    kpi_card(
        "🌡️ Inflasi",
        f"{nilai:.2f}%" if nilai is not None else "N/A",
        "Data inflasi terakhir tersedia"
    )

with d:
    nilai = devisa_valid.iloc[-1] if not devisa_valid.empty else None
    kpi_card(
        "🏦 Cadangan devisa",
        f"${nilai:,.0f} juta" if nilai is not None else "N/A",
        "Data devisa terakhir tersedia"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# TAB DASHBOARD
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Tren Kurs",
    "🔎 Analisis Faktor",
    "🧮 Korelasi",
    "🗃️ Dataset",
    "🤖 Prediksi ML"
])

with tab1:
    st.markdown(
        '<div class="section-title">Perkembangan Kurs USD/IDR</div>',
        unsafe_allow_html=True
    )

    fig = px.line(
        filtered,
        x="period",
        y="kurs_usdidr_rata_rata",
        markers=True,
        labels={
            "period": "Periode",
            "kurs_usdidr_rata_rata": "Rata-rata kurs (Rp)"
        }
    )
    fig.update_traces(
        line=dict(color=CYAN, width=3),
        marker=dict(size=5, color=BLUE),
        hovertemplate="Periode: %{x|%b %Y}<br>Kurs: Rp %{y:,.2f}<extra></extra>"
    )
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown("#### Ringkasan perubahan kurs")
    kurs_tren = filtered.dropna(
        subset=["kurs_usdidr_rata_rata"]
    )
    if len(kurs_tren) >= 2:
        awal = kurs_tren["kurs_usdidr_rata_rata"].iloc[0]
        akhir = kurs_tren["kurs_usdidr_rata_rata"].iloc[-1]
        perubahan = (akhir - awal) / awal * 100 if awal != 0 else 0

        st.write(
            f"Kurs berubah dari **Rp {awal:,.2f}** menjadi "
            f"**Rp {akhir:,.2f}**, dengan perubahan sebesar "
            f"**{perubahan:+.2f}%** pada rentang terpilih."
        )
        st.caption(
            "Persentase positif berarti nilai kurs USD/IDR naik "
            "(lebih banyak rupiah per dolar), bukan berarti rupiah menguat."
        )
    else:
        st.info("Belum cukup data untuk menghitung perubahan kurs.")

with tab2:
    st.markdown(
        '<div class="section-title">Faktor Ekonomi dan Nilai Tukar</div>',
        unsafe_allow_html=True
    )

    kiri, kanan = st.columns(2)

    with kiri:
        fig = px.scatter(
            filtered,
            x="inflasi_persen",
            y="kurs_usdidr_rata_rata",
            color="inflasi_persen",
            color_continuous_scale="Teal",
            hover_data={"period": True},
            labels={
                "inflasi_persen": "Inflasi (%)",
                "kurs_usdidr_rata_rata": "Kurs USD/IDR (Rp)"
            },
            title="Inflasi vs Kurs"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with kanan:
        fig = px.scatter(
            filtered,
            x="cadangan_devisa_usd_juta",
            y="kurs_usdidr_rata_rata",
            color="cadangan_devisa_usd_juta",
            color_continuous_scale="Blues",
            hover_data={"period": True},
            labels={
                "cadangan_devisa_usd_juta": "Cadangan devisa (juta USD)",
                "kurs_usdidr_rata_rata": "Kurs USD/IDR (Rp)"
            },
            title="Cadangan Devisa vs Kurs"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.caption(
        "Setiap titik mewakili satu periode bulanan. "
        "Pola sebaran menunjukkan hubungan, bukan otomatis membuktikan sebab-akibat."
    )

with tab3:
    st.markdown(
        '<div class="section-title">Matriks Korelasi</div>',
        unsafe_allow_html=True
    )

    kolom = [
        "kurs_usdidr_rata_rata",
        "inflasi_persen",
        "cadangan_devisa_usd_juta"
    ]
    nama = {
        "kurs_usdidr_rata_rata": "Kurs USD/IDR",
        "inflasi_persen": "Inflasi",
        "cadangan_devisa_usd_juta": "Cadangan devisa"
    }

    corr_data = filtered[kolom].dropna()
    if len(corr_data) >= 2:
        corr = corr_data.corr().rename(
            index=nama, columns=nama
        )

        fig = px.imshow(
            corr,
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Korelasi Pearson Antarvariabel"
        )
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

        st.caption(
            "Korelasi berkisar dari -1 sampai +1. "
            "Nilai mendekati nol menunjukkan hubungan linear yang lemah. "
            "Korelasi tidak membuktikan hubungan sebab-akibat."
        )
    else:
        st.warning("Data lengkap belum cukup untuk menghitung korelasi.")

with tab4:
    st.markdown(
        '<div class="section-title">Dataset Gabungan</div>',
        unsafe_allow_html=True
    )

    tampilan = filtered.rename(columns={
        "period": "Periode",
        "kurs_usdidr_rata_rata": "Kurs USD/IDR",
        "inflasi_persen": "Inflasi (%)",
        "cadangan_devisa_usd_juta": "Cadangan Devisa (Juta USD)"
    })

    st.dataframe(
        tampilan,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Unduh data CSV",
        data=tampilan.to_csv(index=False).encode("utf-8"),
        file_name="dataset_gabungan_usdidr.csv",
        mime="text/csv"
    )

# =========================
# TAB 5: PREDIKSI ML
# =========================
with tab5:
    st.markdown(
        '<div class="section-title">Prediksi Kurs USD/IDR dengan Linear Regression</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Masukkan nilai inflasi, cadangan devisa, dan kurs bulan sebelumnya. "
        "Model menggunakan fitur yang tersimpan di `fitur_model.pkl`. "
        "Hasil ini merupakan estimasi model, bukan nilai kurs resmi."
    )

    try:
        model_ml, fitur_model = load_ml_assets()
    except Exception as e:
        st.error(
            "Model belum dapat dimuat. Pastikan `model_kurs.pkl` dan "
            "`fitur_model.pkl` berada di folder yang sama dengan app.py, "
            "serta versi scikit-learn kompatibel."
        )
        st.code(str(e))
        st.stop()

    fitur_diharapkan = [
        "inflasi_persen",
        "cadangan_devisa_usd_juta",
        "kurs_lag_1"
    ]
    if list(fitur_model) != fitur_diharapkan:
        st.error(
            "Urutan/nama fitur di fitur_model.pkl tidak sesuai dengan "
            "fitur yang digunakan pada dashboard. Periksa kembali file fitur."
        )
        st.stop()

    # Default input memakai observasi terakhir yang tersedia pada data terpilih.
    data_prediksi = data.sort_values("period").copy()
    inflasi_default = data_prediksi["inflasi_persen"].dropna()
    devisa_default = data_prediksi["cadangan_devisa_usd_juta"].dropna()
    kurs_default = data_prediksi["kurs_usdidr_rata_rata"].dropna()

    if inflasi_default.empty or devisa_default.empty or kurs_default.empty:
        st.warning("Data belum lengkap untuk mengisi nilai awal prediksi. Masukkan nilai secara manual.")
        inflasi_awal = 0.0
        devisa_awal = 0.0
        kurs_lag_awal = 0.0
    else:
        inflasi_awal = float(inflasi_default.iloc[-1])
        devisa_awal = float(devisa_default.iloc[-1])
        kurs_lag_awal = float(kurs_default.iloc[-1])

    kiri, tengah, kanan = st.columns(3)
    with kiri:
        input_inflasi = st.number_input(
            "Inflasi (%)", value=inflasi_awal, format="%.4f", help="Gunakan definisi inflasi yang sama dengan data training."
        )
    with tengah:
        input_devisa = st.number_input(
            "Cadangan devisa (juta USD)", min_value=0.0, value=max(0.0, devisa_awal), format="%.4f"
        )
    with kanan:
        input_kurs_lag = st.number_input(
            "Kurs bulan sebelumnya (Rp/USD)", min_value=0.0, value=max(0.0, kurs_lag_awal), format="%.4f"
        )

    if st.button("🔮 Hitung prediksi kurs", type="primary", key="btn_prediksi_kurs"):
        input_df = pd.DataFrame([{
            "inflasi_persen": input_inflasi,
            "cadangan_devisa_usd_juta": input_devisa,
            "kurs_lag_1": input_kurs_lag
        }])[fitur_model]
        try:
            hasil_prediksi = float(model_ml.predict(input_df)[0])
            st.success(f"Estimasi kurs USD/IDR: Rp {hasil_prediksi:,.2f} per USD")
            st.caption(
                "Prediksi menggunakan model Linear Regression tersimpan. "
                "Nilai input harus menggunakan satuan dan preprocessing yang sama dengan saat model dilatih."
            )
        except Exception as e:
            st.error("Prediksi gagal dijalankan. Periksa kecocokan fitur dan model.")
            st.code(str(e))


st.divider()
st.markdown(
    "<div style='text-align:center;color:#8197BD;font-size:12px;'>"
    "USD/IDR Economic Intelligence · Data Science Project · "
    "Universitas Halu Oleo"
    "</div>",
    unsafe_allow_html=True
)
