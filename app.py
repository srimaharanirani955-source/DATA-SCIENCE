import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="USD/IDR | Economic Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONFIGURASI WARNA
# ============================================================
BG = "#0B1120"
PANEL = "#111C31"
CYAN = "#38D9F9"
BLUE = "#5687FF"
TEXT = "#E8F0FF"
MUTED = "#8EA2C8"
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.stApp { background: #0B1120; color: #E8F0FF; }
[data-testid="stSidebar"] { background: #101A2D; border-right: 1px solid #243653; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { color: #E8F0FF !important; }
p, label, span { color: #D4DFF5; }
.hero { background: linear-gradient(120deg, #152B4A, #123C53); border: 1px solid #28556E; padding: 28px; border-radius: 18px; margin-bottom: 22px; }
.hero-title { font-size: 30px; font-weight: 800; color: #FFFFFF; margin-bottom: 7px; }
.hero-sub { color: #A9C5E7; font-size: 14px; }
.kpi { background: #111C31; border: 1px solid #263A59; border-radius: 15px; padding: 20px; min-height: 132px; }
.kpi-label { color: #9FB3D6; font-size: 13px; margin-bottom: 12px; }
.kpi-value { color: #38D9F9; font-size: 25px; font-weight: 800; overflow-wrap: anywhere; }
.kpi-note { color: #8197BD; font-size: 12px; margin-top: 8px; }
.section-title { font-size: 21px; font-weight: 750; margin-top: 16px; margin-bottom: 10px; color: #E8F0FF; }
div[data-testid="stPlotlyChart"] { background: #111C31; border: 1px solid #263A59; border-radius: 14px; padding: 8px; }
div[data-testid="stDataFrame"] { border: 1px solid #263A59; border-radius: 12px; }
div.stButton > button { border-radius: 9px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BACA DATA DARI FILE HASIL NOTEBOOK (TANPA MYSQL)
# ============================================================
@st.cache_data

def load_data_from_bytes(file_bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    return prepare_data(df)


def prepare_data(df):
    df = df.copy()
    # Mendukung nama kolom yang digunakan pada notebook.
    if "period" not in df.columns:
        for candidate in ["Periode", "tanggal", "date", "month"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "period"})
                break
    required = ["kurs_usdidr_rata_rata", "inflasi_persen", "cadangan_devisa_usd_juta"]
    aliases = {
        "Kurs USD/IDR": "kurs_usdidr_rata_rata",
        "Inflasi (%)": "inflasi_persen",
        "Cadangan Devisa (Juta USD)": "cadangan_devisa_usd_juta",
        "Cadangan Devisa": "cadangan_devisa_usd_juta",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Kolom berikut belum ada di CSV: " + ", ".join(missing))
    if "period" not in df.columns:
        raise ValueError("Kolom periode tidak ditemukan. Pastikan CSV memiliki kolom 'period'.")

    # Periode pada notebook bisa berupa datetime atau teks YYYY-MM.
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    for col in required + (["kurs_lag_1"] if "kurs_lag_1" in df.columns else []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    if "kurs_lag_1" not in df.columns:
        df["kurs_lag_1"] = df["kurs_usdidr_rata_rata"].shift(1)
    return df


@st.cache_resource

def load_model_files():
    model_path = BASE_DIR / "model_kurs.pkl"
    features_path = BASE_DIR / "fitur_model.pkl"
    if not model_path.exists() or not features_path.exists():
        return None, None, f"File model tidak ditemukan. Letakkan model_kurs.pkl dan fitur_model.pkl di folder yang sama dengan app.py."
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features, None
    except Exception as exc:
        return None, None, str(exc)


def style_fig(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=25, r=25, t=55, b=25),
        title_font=dict(size=16, color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor="#253550", zerolinecolor="#253550"),
        yaxis=dict(gridcolor="#253550", zerolinecolor="#253550"),
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


# ============================================================
# BACA DATASET CSV SECARA OTOMATIS (TANPA MYSQL / UPLOAD)
# ============================================================
CSV_PATH = BASE_DIR / "dataset_gabungan.csv"

@st.cache_data

def load_local_csv(path_string, modified_time):
    # modified_time membuat cache diperbarui saat file CSV berubah.
    df = pd.read_csv(path_string)
    return prepare_data(df)

st.sidebar.markdown("## 📈 ECONOMIC INTELLIGENCE")
st.sidebar.caption("USD/IDR • Indonesia")
st.sidebar.divider()
st.sidebar.markdown("### Sumber data")
st.sidebar.write("Dataset dibaca otomatis dari `dataset_gabungan.csv` di folder aplikasi.")
st.sidebar.caption("Tidak menggunakan MySQL, API, atau unggah file manual.")

if not CSV_PATH.exists():
    st.error("File dataset_gabungan.csv belum ditemukan.")
    st.markdown(
        "Simpan file **dataset_gabungan.csv** di folder yang sama dengan **app.py**, "
        "lalu jalankan ulang aplikasi. Nama file harus persis sama."
    )
    st.code(str(CSV_PATH))
    st.stop()

try:
    data = load_local_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
    source_label = "dataset_gabungan.csv (lokal)"
except Exception as exc:
    st.error("Dataset CSV ditemukan, tetapi gagal dibaca atau kolomnya tidak sesuai.")
    st.code(str(exc))
    st.stop()

if data.empty:
    st.error("dataset_gabungan.csv tidak memiliki baris data yang valid.")
    st.stop()

st.sidebar.success(f"Dataset berhasil dibaca: {len(data)} baris")
min_date = data["period"].min().date()
max_date = data["period"].max().date()
st.sidebar.markdown("### Filter periode")
date_range = st.sidebar.date_input(
    "Pilih rentang bulan", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = data[(data["period"].dt.date >= start_date) & (data["period"].dt.date <= end_date)].copy()
else:
    filtered = data.copy()
st.sidebar.divider()
st.sidebar.caption("Sumber: hasil pengolahan notebook")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
  <div class="hero-title">📊 USD/IDR Economic Intelligence</div>
  <div class="hero-sub">Dashboard analisis nilai tukar rupiah berdasarkan inflasi dan cadangan devisa Indonesia, dengan prediksi Linear Regression dari model notebook.</div>
</div>
""", unsafe_allow_html=True)
st.caption(f"Sumber: {source_label} · Periode: {filtered['period'].min():%B %Y} — {filtered['period'].max():%B %Y}" if not filtered.empty else "Tidak ada data pada rentang yang dipilih.")
if filtered.empty:
    st.warning("Tidak ada data pada rentang periode yang dipilih.")
    st.stop()

kurs_valid = filtered["kurs_usdidr_rata_rata"].dropna()
inflasi_valid = filtered["inflasi_persen"].dropna()
devisa_valid = filtered["cadangan_devisa_usd_juta"].dropna()
a, b, c, d = st.columns(4)
with a:
    v = kurs_valid.iloc[-1] if not kurs_valid.empty else None
    kpi_card("💱 Kurs USD/IDR terakhir", f"Rp {v:,.2f}" if v is not None else "N/A", "Nilai terakhir tersedia")
with b:
    v = kurs_valid.mean() if not kurs_valid.empty else None
    kpi_card("📉 Rata-rata kurs", f"Rp {v:,.2f}" if v is not None else "N/A", "Rata-rata periode terpilih")
with c:
    v = inflasi_valid.iloc[-1] if not inflasi_valid.empty else None
    kpi_card("🌡️ Inflasi", f"{v:.2f}%" if v is not None else "N/A", "Nilai terakhir tersedia")
with d:
    v = devisa_valid.iloc[-1] if not devisa_valid.empty else None
    kpi_card("🏦 Cadangan devisa", f"${v:,.0f} juta" if v is not None else "N/A", "Nilai terakhir tersedia")

# ============================================================
# TAB DASHBOARD + PREDIKSI ML
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Tren Kurs", "🔎 Analisis Faktor", "🧮 Korelasi", "🗃️ Dataset", "🤖 Prediksi ML"])

with tab1:
    st.markdown('<div class="section-title">Perkembangan Kurs USD/IDR</div>', unsafe_allow_html=True)
    chart_data = filtered.dropna(subset=["kurs_usdidr_rata_rata"])
    fig = px.line(chart_data, x="period", y="kurs_usdidr_rata_rata", markers=True, labels={"period":"Periode", "kurs_usdidr_rata_rata":"Rata-rata kurs (Rp)"}, title="Kurs USD/IDR bulanan")
    fig.update_traces(line=dict(color=CYAN, width=3), marker=dict(size=5, color=BLUE), hovertemplate="Periode: %{x|%b %Y}<br>Kurs: Rp %{y:,.2f}<extra></extra>")
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    if len(chart_data) >= 2:
        first, last = chart_data["kurs_usdidr_rata_rata"].iloc[0], chart_data["kurs_usdidr_rata_rata"].iloc[-1]
        pct = ((last - first) / first * 100) if first else 0
        st.write(f"Kurs berubah dari **Rp {first:,.2f}** menjadi **Rp {last:,.2f}**, perubahan **{pct:+.2f}%** pada periode terpilih.")
        st.caption("Kenaikan kurs USD/IDR berarti lebih banyak rupiah diperlukan untuk membeli satu dolar AS.")

with tab2:
    st.markdown('<div class="section-title">Faktor Ekonomi dan Nilai Tukar</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        d1 = filtered.dropna(subset=["inflasi_persen", "kurs_usdidr_rata_rata"])
        fig = px.scatter(d1, x="inflasi_persen", y="kurs_usdidr_rata_rata", color="inflasi_persen", color_continuous_scale="Teal", hover_data={"period":True}, labels={"inflasi_persen":"Inflasi (%)", "kurs_usdidr_rata_rata":"Kurs USD/IDR (Rp)"}, title="Inflasi vs Kurs")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        d2 = filtered.dropna(subset=["cadangan_devisa_usd_juta", "kurs_usdidr_rata_rata"])
        fig = px.scatter(d2, x="cadangan_devisa_usd_juta", y="kurs_usdidr_rata_rata", color="cadangan_devisa_usd_juta", color_continuous_scale="Blues", hover_data={"period":True}, labels={"cadangan_devisa_usd_juta":"Cadangan devisa (juta USD)", "kurs_usdidr_rata_rata":"Kurs USD/IDR (Rp)"}, title="Cadangan Devisa vs Kurs")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Pola sebaran menunjukkan hubungan statistik, bukan otomatis membuktikan sebab-akibat.")

with tab3:
    st.markdown('<div class="section-title">Matriks Korelasi Pearson</div>', unsafe_allow_html=True)
    cols = ["kurs_usdidr_rata_rata", "inflasi_persen", "cadangan_devisa_usd_juta", "kurs_lag_1"]
    corr_df = filtered[cols].dropna(how="all")
    corr = corr_df.corr(numeric_only=True)
    rename = {"kurs_usdidr_rata_rata":"Kurs USD/IDR", "inflasi_persen":"Inflasi", "cadangan_devisa_usd_juta":"Cadangan devisa", "kurs_lag_1":"Kurs bulan sebelumnya"}
    corr = corr.rename(index=rename, columns=rename)
    fig = px.imshow(corr, text_auto=".3f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Korelasi antarvariabel")
    st.plotly_chart(style_fig(fig, 450), use_container_width=True)
    st.caption("Korelasi dari -1 sampai +1. Nilai mendekati 0 berarti hubungan linear lemah; korelasi bukan bukti kausalitas.")

with tab4:
    st.markdown('<div class="section-title">Dataset hasil notebook</div>', unsafe_allow_html=True)
    display_df = filtered.rename(columns={"period":"Periode", "kurs_usdidr_rata_rata":"Kurs USD/IDR", "inflasi_persen":"Inflasi (%)", "cadangan_devisa_usd_juta":"Cadangan Devisa (Juta USD)", "kurs_lag_1":"Kurs Bulan Sebelumnya"})
    st.write(f"Jumlah baris pada periode terpilih: **{len(display_df)}**")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh data CSV", data=display_df.to_csv(index=False).encode("utf-8"), file_name="dataset_dashboard_usdidr.csv", mime="text/csv")

with tab5:
    st.markdown('<div class="section-title">Prediksi Kurs USD/IDR — Linear Regression</div>', unsafe_allow_html=True)
    model, features, model_error = load_model_files()
    if model_error:
        st.error(model_error)
    else:
        st.write("Masukkan nilai inflasi, cadangan devisa, dan kurs bulan sebelumnya. Satuan input harus sama dengan yang digunakan saat melatih model di notebook.")
        latest = data.dropna(subset=["kurs_usdidr_rata_rata"]).iloc[-1]
        latest_inflasi = data["inflasi_persen"].dropna()
        latest_devisa = data["cadangan_devisa_usd_juta"].dropna()
        with st.form("form_prediksi"):
            c1, c2, c3 = st.columns(3)
            with c1:
                inflasi_input = st.number_input("Inflasi (%)", value=float(latest_inflasi.iloc[-1]) if not latest_inflasi.empty else 0.0, format="%.4f")
            with c2:
                devisa_input = st.number_input("Cadangan devisa (juta USD)", value=float(latest_devisa.iloc[-1]) if not latest_devisa.empty else 0.0, format="%.4f")
            with c3:
                lag_input = st.number_input("Kurs bulan sebelumnya (Rp/USD)", value=float(latest["kurs_usdidr_rata_rata"]), min_value=0.0, format="%.4f")
            submitted = st.form_submit_button("🔮 Prediksi kurs", use_container_width=True)
        if submitted:
            values = {"inflasi_persen": inflasi_input, "cadangan_devisa_usd_juta": devisa_input, "kurs_lag_1": lag_input}
            try:
                X_input = pd.DataFrame([[values[f] for f in features]], columns=features)
                prediction = float(model.predict(X_input)[0])
                st.success(f"Prediksi kurs USD/IDR: Rp {prediction:,.2f}")
                st.caption("Hasil merupakan estimasi model berdasarkan fitur yang dimasukkan, bukan jaminan nilai kurs aktual.")
            except Exception as exc:
                st.error("Prediksi gagal. Pastikan fitur_model.pkl sesuai dengan model dan kolom pada notebook.")
                st.code(str(exc))
    st.info("Model dibaca dari file PKL yang sudah dilatih di notebook. Dashboard tidak melatih ulang model, sehingga metrik hasil notebook tidak diubah.")

st.divider()
st.markdown("<div style='text-align:center;color:#8197BD;font-size:12px;'>USD/IDR Economic Intelligence · Data Science Project · Universitas Halu Oleo</div>", unsafe_allow_html=True)
