# -*- coding: utf-8 -*-
"""
Prototipo de Validación de Experiencia de Licitación para OKC.
Versión 2.0 - Optimizada bajo el SDK oficial google-genai, Gemini 2.5 Flash
e interfaz con Semáforo de Confianza de Extracción "Human-in-the-loop".
"""

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import datetime
import io
import re
import json
import time
import requests
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================================
# ⚙️ CONFIGURACIÓN GLOBAL DE API KEY (INTEGRACIÓN CON CLIENTE DE GEMINI)
# =========================================================================
DEFAULT_API_KEY = "AIzaSyA6zjXlY9gyL1kqCUI0GxLz5EV4r8ARs_s"

# Configuración de página de Streamlit para OKC
st.set_page_config(
    page_title="OKC - Sistema IA de Validación de Experiencia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar cliente GenAI usando el patrón oficial
def init_gemini_client(api_key):
    """
    Inicializa el cliente de GenAI oficial de Google con reintentos para mitigar errores 503/429.
    """
    try:
        return genai.Client(
            api_key=api_key,
            http_options={"timeout": 120.0},
           # client_options={"transport": "rest"}
        )
    except Exception as e:
        st.error(f"Error inicializando el cliente de Gemini: {e}")
        return None

# Estilos personalizados para la marca OKC
st.markdown("""
<style>
    .main-header {
        font-size:32px !important;
        font-weight: bold;
        color: #0F52BA;
    }
    .accent-card {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0F52BA;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# ETAPA 1: INGESTA DE DATOS Y LIMPIEZA
# -------------------------------------------------------------------------
def clean_and_load_excel(uploaded_file):
    """
    Carga el Excel histórico de OKC y normaliza sus campos clave.
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        req_cols = ["ID_Contrato", "Cliente", "Objeto_Contrato", "Monto_Original", "Moneda", "Fecha_Suscripcion"]
        for col in req_cols:
            if col not in df.columns:
                st.error(f"Falta la columna obligatoria en el Excel: {col}")
                return None
                
        df["Fecha_Suscripcion"] = pd.to_datetime(df["Fecha_Suscripcion"], errors='coerce').dt.date
        
        df["Moneda"] = df["Moneda"].astype(str).str.strip().str.upper()
        df["Moneda"] = df["Moneda"].replace({"SOLES": "PEN", "S/.": "PEN", "S/": "PEN", "DOLARES": "USD", "$": "USD"})
        
        df["Monto_Original"] = df["Monto_Original"].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df["Monto_Original"] = pd.to_numeric(df["Monto_Original"], errors='coerce')
        
        df = df.dropna(subset=["Monto_Original", "Fecha_Suscripcion", "Objeto_Contrato"])
        return df
    except Exception as e:
        st.error(f"Error procesando el Excel: {str(e)}")
        return None

# -------------------------------------------------------------------------
# ETAPA 2: PARSER DE BASES (EXTRACCIÓN SEMÁNTICA CON AUTOCALIFICACIÓN DE CONFIANZA)
# -------------------------------------------------------------------------
def call_gemini_structured_parser(text_context, client):
    """
    Utiliza el SDK oficial google-genai para extraer criterios exigidos y evaluar el nivel de confianza.
    """
    prompt_sistema = (
        "Eres un Auditor Experto en Contrataciones del Estado Peruano (OSCE) y Licitaciones Privadas. "
        "Tu objetivo es leer un fragmento de las bases de una licitación y extraer los criterios de calificación "
        "de la experiencia del postor de manera exacta, ignorando montos correspondientes al presupuesto de la obra. "
        "Debes extraer única y exclusivamente la experiencia y facturación que el postor está obligado a acreditar. "
        "Adicionalmente, debes evaluar de 0 a 100 qué tan explícita, clara y libre de ambigüedades es la información analizada."
    )
    
    contenido_instruccion = (
        "Analiza el siguiente texto extraído de las bases y extrae los criterios de calificación exigidos al postor.\n"
        "Reglas de Normalización:\n"
        "1. Si no se especifica explícitamente la antigüedad de la experiencia, asume 8 años por defecto (norma general OSCE para servicios).\n"
        "2. Si no se especifica el monto mínimo de facturación, asume 0.0.\n"
        "3. Normaliza el campo 'moneda' únicamente a códigos ISO estándar: 'PEN' o 'USD'.\n"
        "4. En 'descripcion_similitud', captura la descripción conceptual completa y limpia de los servicios considerados similares.\n"
        "5. En 'confianza_extraccion', asigna un valor numérico de 0.0 a 100.0 según la claridad técnica del texto provisto.\n\n"
        f"Texto de las bases:\n{text_context}"
    )

    esquema_salida = {
        "type": "OBJECT",
        "properties": {
            "monto_requerido": {"type": "NUMBER"},
            "moneda": {"type": "STRING"},
            "antiguedad_anos": {"type": "INTEGER"},
            "descripcion_similitud": {"type": "STRING"},
            "confianza_extraccion": {"type": "NUMBER"}
        },
        "required": ["monto_requerido", "moneda", "antiguedad_anos", "descripcion_similitud", "confianza_extraccion"]
    }

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contenido_instruccion,
        config=types.GenerateContentConfig(
            system_instruction=prompt_sistema,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=esquema_salida
        )
    )
    return json.loads(response.text)

def parse_bases_pdf(pdf_file, client):
    """
    Parser híbrido con asignación balanceada en caso de caída o fallback técnico.
    """
    extracted_text = ""
    parsed_criteria = {
        "monto_requerido": 0.0,
        "moneda": "PEN",
        "antiguedad_anos": 8,
        "descripcion_similitud": "",
        "confianza_extraccion": 50.0  # El fallback por defecto asume 50% de confianza
    }
    
    try:
        target_window = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                extracted_text += text + "\n"
                
                if any(k in text.upper() for k in ["EXPERIENCIA DEL POSTOR", "EXPERIENCIA DE LOS POSTORES", "REQUISITOS DE CALIFICACIÓN"]):
                    target_window += text + "\n"

        analysis_context = target_window if target_window else extracted_text
        analysis_context = analysis_context[:15000]

        if client is not None:
            try:
                extracted_json = call_gemini_structured_parser(analysis_context, client)
                
                monto = float(extracted_json.get("monto_requerido", 0.0))
                antiguedad = int(extracted_json.get("antiguedad_anos", 8))
                moneda = str(extracted_json.get("moneda", "PEN")).strip().upper()
                if moneda not in ["PEN", "USD"]:
                    moneda = "PEN"
                descripcion = str(extracted_json.get("descripcion_similitud", "")).strip()
                confianza = float(extracted_json.get("confianza_extraccion", 85.0))
                
                json_validado = {
                    "monto_requerido": monto,
                    "moneda": moneda,
                    "antiguedad_anos": antiguedad,
                    "descripcion_similitud": descripcion,
                    "confianza_extraccion": confianza
                }
                parsed_criteria.update(json_validado)
                return parsed_criteria, extracted_text[:5000]
            except Exception as e:
                st.warning(f"⚠️ Error en pipeline nativo de Gemini SDK: {str(e)}. Ejecutando fallback local...")
        
        # Fallback Heurístico Local por Proximidad (Si falla la API o no hay cliente)
        lines = analysis_context.split('\n')
        for i, line in enumerate(lines):
            if "SIMILAR" in line.upper() or "SIMILARES" in line.upper():
                context_chunk = lines[max(0, i-1): min(len(lines), i+8)]
                parsed_criteria["descripcion_similitud"] = " ".join([l.strip() for l in context_chunk])
                break

        regex_moneda_monto = r'(S/\.?|Soles|USD|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        lineas_criticas = [line for line in lines if any(keyword in line.upper() for keyword in ["FACTURACIÓN", "ACUMULADO", "VECES", "EQUIVALENTE"])]
        contexto_financiero = " ".join(lineas_criticas)
        
        monto_matches = re.findall(regex_moneda_monto, contexto_financiero, re.IGNORECASE)
        if monto_matches:
            moneda_detectada, cifra_texto = monto_matches[0]
            parsed_criteria["monto_requerido"] = float(cifra_texto.replace(',', ''))
            parsed_criteria["moneda"] = "USD" if moneda_detectada.upper() in ["USD", "$"] else "PEN"

        antiguedad_match = re.search(r'(?:antigüedad|experiencia)\s+(?:no\s+mayor\s+a|de|hasta)\s+(\d+)\s+años', analysis_context, re.IGNORECASE)
        if antiguedad_match:
            parsed_criteria["antiguedad_anos"] = int(antiguedad_match.group(1))
            
        return parsed_criteria, extracted_text[:5000]
    except Exception as e:
        st.warning(f"Error crítico en proceso de parsing: {str(e)}. Se usará entrada manual.")
        return parsed_criteria, ""

# -------------------------------------------------------------------------
# ETAPA 3: EVALUACIÓN DE SIMILITUD COGNITIVA MEDIANTE SDK NATIVO
# -------------------------------------------------------------------------
def call_gemini_similarity_scorer(target_description, contracts_list, client):
    """
    Consolida objetos contractuales e infiere las métricas conceptuales de similitud mediante la API de Gemini.
    """
    system_prompt = (
        "Eres un evaluador técnico experto en procesos de licitación pública y privada. "
        "Tu labor es comparar objetos de contratos históricos ejecutados con la definición de 'servicio similar' "
        "requerida en las bases, asignando un puntaje conceptual de similitud extremadamente preciso."
    )
    
    contracts_str = ""
    for c in contracts_list:
        contracts_str += f"- [ID: {c['ID_Contrato']}] {c['Objeto_Contrato']}\n"
        
    prompt_instruccion = (
        f"Requisito de 'Servicio Similar' exigido en las bases:\n\"{target_description}\"\n\n"
        "Portafolio de contratos históricos de OKC a evaluar:\n"
        f"{contracts_str}\n\n"
        "Instrucciones de Evaluación:\n"
        "1. Compara cada contrato individualmente frente al requisito similar.\n"
        "2. Asigna un score de similitud (de 0.00 a 1.00).\n"
        "3. Devuelve el resultado en un JSON estructurado bajo el esquema exacto provisto."
    )

    esquema_similitud = {
        "type": "OBJECT",
        "properties": {
            "scores": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "score": {"type": "NUMBER"}
                    },
                    "required": ["id", "score"]
                }
            }
        },
        "required": ["scores"]
    }

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_instruccion,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=esquema_similitud
        )
    )
    return json.loads(response.text)

def calculate_semantic_similarity(target_description, database_df, client):
    """
    Calcula la correspondencia semántica usando el motor de razonamiento de Gemini.
    """
    descriptions = database_df["Objeto_Contrato"].tolist()
    
    if client is not None:
        try:
            progress_bar = st.progress(0, text="Iniciando Razonamiento Semántico con Gemini 2.5 Flash SDK...")
            
            contracts_list = []
            for _, row in database_df.iterrows():
                contracts_list.append({
                    "ID_Contrato": int(row["ID_Contrato"]),
                    "Objeto_Contrato": str(row["Objeto_Contrato"])
                })
            
            progress_bar.progress(30, text="Portafolio empaquetado. Evaluando similitudes cognitivas...")
            
            output_json = call_gemini_similarity_scorer(target_description, contracts_list, client)
            progress_bar.progress(80, text="Mapeando vectores semánticos...")
            
            scores_map = {item["id"]: float(item["score"]) for item in output_json.get("scores", [])}
            database_df["Similitud_Score"] = database_df["ID_Contrato"].apply(lambda idx: scores_map.get(int(idx), 0.0))
            
            progress_bar.progress(100, text="Razonamiento conceptual finalizado.")
        except Exception as e:
            st.warning(f"⚠️ Error en API de Gemini para similitud: {str(e)}. Activando motor estadístico clásico...")
            run_fallback_tfidf(target_description, database_df, descriptions)
    else:
        run_fallback_tfidf(target_description, database_df, descriptions)
        
    return database_df

def run_fallback_tfidf(target, df, corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=None)
    all_texts = [target] + corpus
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    df["Similitud_Score"] = scores[0]

# -------------------------------------------------------------------------
# ETAPA 4: REGLAS DE NEGOCIO (CONEXIÓN API TIPO CAMBIO SUNAT Y OPTIMIZACIÓN)
# -------------------------------------------------------------------------
@st.cache_data(ttl=1200)
def fetch_sunat_exchange_rate(date_str):
    url = f"https://api.apis.net.pe/v1/tipo-cambio-sunat?fecha={date_str}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data.get("venta", 3.75))
    except Exception:
        pass
    return 3.75

def apply_business_rules_and_optimize(df, target_monto, target_moneda, max_years, ref_date):
    """
    Filtra por antigüedad, convierte monedas dinámicamente y optimiza el portafolio.
    """
    df["Antiguedad_Dias"] = df["Fecha_Suscripcion"].apply(lambda x: (ref_date - x).days)
    df["Valido_Tiempo"] = df["Antiguedad_Dias"] <= (max_years * 365)
    
    df_valid = df[df["Valido_Tiempo"] == True].copy()
    
    def convert_amount(row):
        monto = row["Monto_Original"]
        moneda_org = row["Moneda"]
        fecha_con = str(row["Fecha_Suscripcion"])
        
        if moneda_org == target_moneda:
            return monto
            
        tipo_cambio = fetch_sunat_exchange_rate(fecha_con)
        
        if moneda_org == "USD" and target_moneda == "PEN":
            return monto * tipo_cambio
        elif moneda_org == "PEN" and target_moneda == "USD":
            return monto / tipo_cambio
        return monto

    if not df_valid.empty:
        with st.spinner("Consultando Tipo de Cambio Histórico de SUNAT para los contratos..."):
            df_valid["Monto_Convertido"] = df_valid.apply(convert_amount, axis=1)
    else:
        df_valid["Monto_Convertido"] = pd.Series(dtype=float)

    df_optimized = pd.DataFrame()
    if not df_valid.empty:
        df_sorted = df_valid.sort_values(by="Similitud_Score", ascending=False)
        
        monto_acumulado = 0.0
        selected_indices = []
        UBRAL_MINIMO = 0.15 # Corregido error tipográfico del script base
        
        for idx, row in df_sorted.iterrows():
            if row["Similitud_Score"] >= UBRAL_MINIMO:
                selected_indices.append(idx)
                monto_acumulado += row["Monto_Convertido"]
                if monto_acumulado >= (target_monto * 1.1):
                    break
        
        df_optimized = df_sorted.loc[selected_indices]
        
    return df_valid, df_optimized

# -------------------------------------------------------------------------
# ETAPA 5: INTERFAZ DE USUARIO (VISTA STREAMLIT CON AUDITORÍA "HUMAN-IN-THE-LOOP")
# -------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">OKC - Inteligencia de Acreditación de Licitaciones</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.header("📥 Ingesta de Archivos")
    excel_file = st.sidebar.file_uploader("1. Registro de Experiencia OKC (Excel)", type=["xlsx"])
    pdf_file = st.sidebar.file_uploader("2. Bases de la Licitación (PDF)", type=["pdf"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuración del Cerebro IA")
    
    gemini_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=DEFAULT_API_KEY,
        type="password", 
        help="Clave API de Google para análisis estructurado de PDF y razonamiento de similitud"
    )
    
    ref_date = st.sidebar.date_input("Fecha de Convocatoria del Proceso", datetime.date.today())
    
    # Inicialización centralizada del cliente de procesamiento
    client = init_gemini_client(gemini_key) if gemini_key.strip() else None

    if "okc_db" not in st.session_state:
        if excel_file is None:
            st.info("👋 Por favor, inicie cargando el archivo Excel histórico de OKC en el panel lateral.")
            if st.button("Cargar Datos de Demostración para Prueba Rápida"):
                demo_data = {
                    "ID_Contrato": [1, 2, 3, 4, 5],
                    "Cliente": ["MINERA LAS BAMBAS", "PRONABEC", "MUNICIPALIDAD DE LIMA", "SAB MILLER", "MINISTERIO DE EDUCACIÓN"],
                    "Objeto_Contrato": [
                        "Soporte técnico integral de infraestructura tecnológica operacional, servidores y redes 24x7 en operaciones críticas de campo minero.",
                        "Adquisición de equipamiento informático compuesto por 500 computadoras portátiles tipo laptop y servicio de puesta en marcha.",
                        "Implementación de centro de datos, cableado estructurado, fibra óptica, conectividad y seguridad de red institucional.",
                        "Suministro de dispositivos de usuario final, tablets corporativas y soporte informático a nivel nacional.",
                        "Aprovisionamiento e instalación de equipamiento tecnológico interactivo de computación y servidores para colegios rurales."
                    ],
                    "Monto_Original": [1200000.0, 500000.0, 850000.0, 300000.0, 1500000.0],
                    "Moneda": ["USD", "PEN", "PEN", "USD", "PEN"],
                    "Fecha_Suscripcion": [datetime.date(2021, 6, 15), datetime.date(2023, 2, 10), datetime.date(2022, 11, 1), datetime.date(2024, 1, 20), datetime.date(2018, 5, 5)]
                }
                st.session_state["okc_db"] = pd.DataFrame(demo_data)
                st.success("¡Datos de prueba cargados exitosamente!")
        else:
            db_df = clean_and_load_excel(excel_file)
            if db_df is not None:
                st.session_state["okc_db"] = db_df
                st.success(f"Base de datos de OKC cargada correctamente.")
                
    # Flujo ejecutable principal
    if "okc_db" in st.session_state and pdf_file is not None:
        st.subheader("📑 Resultados del Análisis de Acreditación Inteligente")
        with st.spinner("Ejecutando Pipeline Semántico en el PDF con Gemini..."):
            criterios, partial_text = parse_bases_pdf(pdf_file, client)
            
        # --- 🔍 VISUALIZACIÓN DE CRITERIOS CON SEMÁFORO DE CONFIANZA ---
        with st.expander("🔍 Auditoría de Criterios Extraídos (IA)", expanded=True):
            confianza = criterios.get("confianza_extraccion", 50.0) / 100.0
            
            if confianza >= 0.85:
                st.success(f"🎯 **Confianza Alta ({confianza*100:.0f}%)**: Los datos coinciden plenamente con la estructura estándar detectada.")
            elif confianza >= 0.60:
                st.warning(f"⚠️ **Confianza Media ({confianza*100:.0f}%)**: El texto de las bases presenta ambigüedades. Verifique los campos.")
            else:
                st.error(f"🚨 **Confianza Baja ({confianza*100:.0f}%)**: Extracción por contingencia heurística. Se requiere revisión humana obligatoria.")
            
            st.progress(confianza)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                criterios['monto_requerido'] = st.number_input("Monto Requerido", value=float(criterios['monto_requerido']))
            with col_b:
                criterios['moneda'] = st.selectbox("Moneda", ["PEN", "USD"], index=0 if criterios['moneda'] == "PEN" else 1)
            with col_c:
                criterios['antiguedad_anos'] = st.number_input("Antigüedad (Años)", value=int(criterios['antiguedad_anos']))
            with col_d:
                st.info(f"**Detector IA:** {'Activo' if client else 'Fallback'}")
            
            criterios['descripcion_similitud'] = st.text_area("Descripción de Similitud Técnica:", value=criterios['descripcion_similitud'], height=100)
            st.caption("Nota: Modificar cualquiera de los campos superiores recalculará dinámicamente el portafolio óptimo.")
        # ----------------------------------------------------------------------------
        
        # Continuar con la métrica final utilizando la estructura del expander (fuente única de verdad)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="💰 Monto Exigido (Final)", value=f"{criterios['moneda']} {criterios['monto_requerido']:,}")
        with col2:
            st.metric(label="📅 Antigüedad Límite", value=f"{criterios['antiguedad_anos']} Años")
        with col3:
            st.metric(label="🎯 Motor de Similitud", value="Gemini Cognitivo" if client else "TF-IDF Estadístico")
            
        # Ejecutar análisis semántico sobre la base de datos
        df_evaluado = calculate_semantic_similarity(criterios['descripcion_similitud'], st.session_state["okc_db"].copy(), client)
        
        # Optimizar portafolio basándose en las variables interactivas auditadas
        df_valid, df_opt = apply_business_rules_and_optimize(
            df_evaluado, criterios['monto_requerido'], criterios['moneda'], criterios['antiguedad_anos'], ref_date
        )
        
        st.markdown("### 🏆 Portafolio Óptimo Sugerido para Presentación")
        if not df_opt.empty:
            st.dataframe(df_opt[["ID_Contrato", "Cliente", "Objeto_Contrato", "Similitud_Score", "Monto_Convertido"]], use_container_width=True)
            monto_total_acreditado = df_opt["Monto_Convertido"].sum()
            
            if monto_total_acreditado >= criterios['monto_requerido']:
                st.success(f"✅ **APTO:** Se cubre la cuota exigida en las bases. Total acumulado óptimo: {criterios['moneda']} {monto_total_acreditado:,.2f}")
            else:
                st.error(f"❌ **NO APTO:** Los contratos similares acumulados suman {criterios['moneda']} {monto_total_acreditado:,.2f}, insuficiente para calificar.")
        else:
            st.warning("No se encontraron contratos históricos que cumplan con los criterios mínimos de coincidencia técnica o ventanas de antigüedad.")

if __name__ == '__main__':
    main()