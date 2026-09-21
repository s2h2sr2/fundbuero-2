import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from datetime import date
import io

# ─────────────────────────────────────────
# Seitenkonfiguration
# ─────────────────────────────────────────

st.set_page_config(page_title="Das Fundbüro", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────
# Globales CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 0rem; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 3rem 3rem 2.5rem 3rem;
        border-radius: 0 0 24px 24px;
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: white;
        font-size: 4rem;
        margin: 0;
        line-height: 1.1;
    }
    .hero span.rot { color: #e94560; }
    .hero p {
        color: #aaaacc;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .info-card {
        background-color: #1e1e2e;
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        height: 100%;
    }
    .info-card h3 { color: #e94560; margin-top: 0; }
    .info-card p  { color: #ccccdd; }

    .filter-bar {
        background-color: #f5f5f5;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }

    .gegenstand-karte {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        height: 100%;
    }
    .gegenstand-karte p {
        margin: 0.2rem 0;
        font-size: 0.85rem;
        color: #555;
    }
    .gegenstand-karte strong {
        font-size: 1rem;
        color: #1a1a2e;
    }

    .nav-bar {
        display: flex;
        gap: 1rem;
        padding: 0.8rem 2rem;
        background-color: #1a1a2e;
        border-radius: 0 0 12px 12px;
        margin-bottom: 0;
    }
    .nav-bar span {
        color: #aaaacc;
        font-size: 1rem;
        cursor: pointer;
    }
    .nav-bar span.aktiv {
        color: #e94560;
        font-weight: bold;
        border-bottom: 2px solid #e94560;
        padding-bottom: 2px;
    }

    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Modell laden
# ─────────────────────────────────────────

MODEL_PATH = "models/dein_model.h5"
KATEGORIEN = ["Hoodie", "Hose", "Flasche", "Schuhe"]
IMG_SIZE = (224, 224)

from tensorflow.keras.layers import DepthwiseConv2D

class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modell nicht gefunden unter: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
    )
    return model

model = load_model()

# ─────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────

def klassifiziere_bild(image: Image.Image) -> tuple:
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    vorhersage = model.predict(img_array)
    index = int(np.argmax(vorhersage))
    konfidenz = float(np.max(vorhersage)) * 100
    return KATEGORIEN[index], konfidenz

def bild_zu_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────

if "gegenstaende" not in st.session_state:
    st.session_state.gegenstaende = []

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────

st.markdown("""
<div class='hero'>
    <h1>Das <span class='rot'>Fund</span>büro 🔍</h1>
    <p>Deine Schulplattform für verlorene und gefundene Gegenstände.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INFO-KARTEN: Was ist das Fundbüro?
# ─────────────────────────────────────────

st.markdown("### 📌 Was ist das Fundbüro?")
k1, k2 = st.columns(2, gap="large")

with k1:
    st.markdown("""
    <div class='info-card'>
        <h3>📦 Hast du was gefunden?</h3>
        <p>Hier kannst du alles, was du findest, hochladen,
        damit Leute ihr Eigentum wiederfinden können.</p>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class='info-card'>
        <h3>🔍 Hast du was verloren?</h3>
        <p>Hiermit kannst du deinen verlorenen Gegenstand suchen.
        Mit hilfreichen Filtern geht es ganz fix.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────
# TABS: Finden / Suchen
# ─────────────────────────────────────────

tab_suchen, tab_finden = st.tabs(["🔍 Suchen", "📦 Gegenstand eintragen"])

# ══════════════════════════════════════════
# TAB: SUCHEN
# ══════════════════════════════════════════

with tab_suchen:

    # Filterleiste
    st.markdown("### 🎛️ Filter")
    f1, f2, f3, f4 = st.columns(4, gap="medium")

    with f1:
        filter_kategorie = st.selectbox("Kategorie", ["Alle"] + KATEGORIEN)
    with f2:
        filter_farbe = st.text_input("Farbe", placeholder="z.B. Rot")
    with f3:
        filter_groesse = st.selectbox("Größe", ["Alle", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"])
    with f4:
        filter_material = st.text_input("Material", placeholder="z.B. Leder")

    st.markdown("---")

    # Ergebnisse filtern
    ergebnisse = st.session_state.gegenstaende

    if filter_kategorie != "Alle":
        ergebnisse = [e for e in ergebnisse if e["kategorie"] == filter_kategorie]
    if filter_farbe.strip():
        ergebnisse = [e for e in ergebnisse if filter_farbe.strip().lower() in e["farbe"].lower()]
    if filter_groesse != "Alle":
        ergebnisse = [e for e in ergebnisse if e["groesse"] == filter_groesse]
    if filter_material.strip():
        ergebnisse = [e for e in ergebnisse if filter_material.strip().lower() in e["material"].lower()]

    # Ergebnisse als Kacheln (3 pro Zeile)
    if not ergebnisse:
        st.info("ℹ️ Keine Gegenstände gefunden. Passe deine Filter an oder trage zuerst etwas ein!")
    else:
        st.markdown(f"**{len(ergebnisse)} Gegenstand/Gegenstände gefunden:**")
        spalten_anzahl = 3
        for zeile_start in range(0, len(ergebnisse), spalten_anzahl):
            zeilen_eintraege = ergebnisse[zeile_start:zeile_start + spalten_anzahl]
            cols = st.columns(spalten_anzahl, gap="medium")
            for col, eintrag in zip(cols, zeilen_eintraege):
                with col:
                    with st.container(border=True):
                        st.image(eintrag["bild"], use_container_width=True)
                        st.markdown(f"**{eintrag['kategorie']}**")
                        st.markdown(f"🎨 Farbe: {eintrag['farbe'] or '–'}")
                        st.markdown(f"📐 Größe: {eintrag['groesse']}")
                        st.markdown(f"🧵 Material: {eintrag['material'] or '–'}")
                        st.markdown(f"📅 Datum: {eintrag['datum']}")
                        idx = st.session_state.gegenstaende.index(eintrag)
                        if st.button("🗑️ Entfernen", key=f"del_{idx}"):
                            st.session_state.gegenstaende.pop(idx)
                            st.rerun()

# ══════════════════════════════════════════
# TAB: GEGENSTAND EINTRAGEN
# ══════════════════════════════════════════

with tab_finden:
    st.markdown("### 📦 Gegenstand hochladen")

    upload_col, info_col = st.columns([1, 1], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Bild hochladen (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

            with st.spinner("🤖 KI analysiert das Bild..."):
                kategorie, konfidenz = klassifiziere_bild(image)

            st.success(f"✅ Erkannte Kategorie: **{kategorie}** ({konfidenz:.1f}% sicher)")

    with info_col:
        if uploaded_file is not None:
            st.markdown("### 📝 Details eingeben")
            farbe    = st.text_input("Farbe des Gegenstands", placeholder="z.B. Blau")
            groesse  = st.selectbox("Größe", ["–", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"])
            material = st.text_input("Material (optional)", placeholder="z.B. Baumwolle")
            funddatum = st.date_input("Funddatum", value=date.today())

            if st.button("💾 Gegenstand eintragen", use_container_width=True, type="primary"):
                eintrag = {
                    "bild":      bild_zu_bytes(image),
                    "kategorie": kategorie,
                    "farbe":     farbe,
                    "groesse":   groesse,
                    "material":  material,
                    "datum":     str(funddatum),
                }
                st.session_state.gegenstaende.append(eintrag)
                st.success("🎉 Gegenstand wurde eingetragen! Wechsle zum Tab 'Suchen', um ihn zu sehen.")
        else:
            st.info("⬅️ Lade zuerst ein Bild hoch, um die Details einzugeben.")
