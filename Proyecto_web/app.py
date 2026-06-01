import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Importar tus módulos personalizados
from modules.image_loader import load_image
from modules.preprocessing import extract_rois, grayscale_and_blur
from modules.segmentation import segment_upper, segment_lower
from modules.morphology import clean_mask
from modules.counting import count_panels
from modules.evaluation import evaluate

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA WEB
# =========================================================
st.set_page_config(
    page_title="Contador de Paneles Solares - Unimagdalena",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Proyecto de Segmentación de Imágenes de Dron")
st.markdown("""
Esta aplicación web presenta el flujo de procesamiento paso a paso, modelando la imagen aérea como una **señal discreta bidimensional**, aplicando técnicas de filtrado espacial, operaciones no lineales de amplitud y morfología matemática.
""")

# Justificación técnica (Teoría de señales)
with st.expander("📘 Fundamentos de Señales Espaciales (Justificación Técnica)"):
    st.markdown("""
    **Marco Teórico - Procesamiento de Señales I**
    
    Para este análisis sobre el edificio docente de la Universidad del Magdalena, la imagen se aborda formalmente como una **señal discreta bidimensional**, donde cada píxel es una muestra espacial.
    
    * **Filtrado Espacial (Pasa-bajas):** Antes de la segmentación, se aplica un suavizado Gaussiano. En el dominio de las frecuencias, esto actúa como un filtro pasa-bajas que atenúa el ruido de alta frecuencia (variaciones bruscas), estabilizando la señal.
    * **Umbralización (Operación no lineal):** La segmentación es una operación no lineal sobre la amplitud de la señal espacial. Se utilizó una estrategia híbrida (adaptativa y global) para contrarrestar los cambios de iluminación y sombras.
    * **Morfología Matemática:** Funciona como un filtro espacial no lineal de post-procesamiento para rellenar discontinuidades y separar frecuencias espaciales adyacentes (paneles muy juntos).
    
    *Desarrollado por: Camilo Cantillo, Luis Mercado*
    """)

# =========================================================
# BARRA LATERAL (SIDEBAR) 
# =========================================================
st.sidebar.header("⚙️ Panel de Control")

opcion_imagen = st.sidebar.selectbox(
    "1. Selección de Imagen:",
    ["DJI_0612.JPG (Predeterminada)", "DJI_0613.JPG"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nombre_archivo = "DJI_0613.JPG" if "0613" in opcion_imagen else "DJI_0612.JPG"
image_path = os.path.join(BASE_DIR, "images", nombre_archivo)

img_ready = os.path.exists(image_path)

if img_ready:
    st.sidebar.success(f"⚡ Procesando en tiempo real: {nombre_archivo}")
else:
    st.sidebar.error(f"No se encontró '{nombre_archivo}'")

st.sidebar.divider()
st.sidebar.subheader("2. Ajuste de Umbrales Globales")
threshold_sup = st.sidebar.slider("Umbral ROI Superior (Sol)", 0, 255, 81)
threshold_inf = st.sidebar.slider("Umbral ROI Inferior", 0, 255, 90)
area_min = st.sidebar.number_input("Área Mínima de Panel (px)", 100, 10000, 4000, 100)

st.sidebar.divider()
st.sidebar.subheader("3. Filtro Adaptativo (Sombra)")
block_size = st.sidebar.slider("Block Size (Impar)", 3, 99, 35, 2)
c_value = st.sidebar.slider("Constante (C)", -20, 50, 12, 1)

conteo_real_sup = 36
conteo_real_inf = 36

# =========================================================
# CUERPO PRINCIPAL DE LA PÁGINA
# =========================================================

if img_ready:
    try:
        # 1. CARGA DE IMAGEN 
        img_rgb = load_image(image_path)
        
        # Generar Imagen con ROIs delimitadas (Fig. 3 del documento)
        img_con_rois = img_rgb.copy()
        # Coordenadas exactas extraídas del marco teórico
        cv2.rectangle(img_con_rois, (1120, 80), (2925, 445), (0, 255, 0), 10)  # ROI Superior
        cv2.rectangle(img_con_rois, (940, 1634), (3140, 2154), (0, 255, 0), 10) # ROI Inferior

        # 2. PREPROCESAMIENTO
        roi_sup, roi_inf = extract_rois(img_rgb)
        gray_sup, blur_sup = grayscale_and_blur(roi_sup)
        gray_inf, blur_inf = grayscale_and_blur(roi_inf)

        # 3. SEGMENTACIÓN
        corte = 510
        sombra = blur_sup[:, :corte]
        sol = blur_sup[:, corte:]
        
        _, mascara_sol = cv2.threshold(sol, threshold_sup, 255, cv2.THRESH_BINARY_INV)
        mascara_sombra = cv2.adaptiveThreshold(
            sombra, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
        mask_sup = np.hstack((mascara_sombra, mascara_sol))
        _, mask_inf = cv2.threshold(blur_inf, threshold_inf, 255, cv2.THRESH_BINARY_INV)

        # 4. MORFOLOGÍA
        mask_sup_clean = clean_mask(mask_sup)
        mask_inf_clean = clean_mask(mask_inf)

        # 5. CONTEO
        paneles_sup = count_panels(mask_sup_clean, area_minima=area_min)
        paneles_inf = count_panels(mask_inf_clean, area_minima=area_min)
        conteo_sup = len(paneles_sup)
        conteo_inf = len(paneles_inf)

        # 6. EVALUACIÓN
        error_abs_sup, error_pct_sup = evaluate(conteo_real_sup, conteo_sup)
        error_abs_inf, error_pct_inf = evaluate(conteo_real_inf, conteo_inf)

        # === RENDERIZADO VISUAL POR ETAPAS (ALINEADO AL DOCUMENTO) ===
        
        # PESTAÑAS ESTRUCTURALES
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Adquisición y Preprocesamiento (Figs. 3-5)",
            "2. Segmentación Híbrida (Fig. 6)",
            "3. Morfología Matemática (Fig. 7)",
            "4. Resultados y Conteo (Fig. 8)"
        ])

        with tab1:
            st.subheader("Delimitación de Regiones de Interés (ROIs)")
            st.image(img_con_rois, caption="Fig 3. Imagen original con ROIs segmentadas espacialmente", use_container_width=True)
            
            st.divider()
            st.subheader("Transformación y Filtrado Espacial Pasa-Bajas")
            
            # ROI Superior
            st.markdown("**Región Superior (Techo Edificio Docente)**")
            col_rgb_sup, col_gray_sup, col_blur_sup = st.columns(3)
            col_rgb_sup.image(roi_sup, caption="(a) ROI Superior RGB", use_container_width=True)
            col_gray_sup.image(gray_sup, caption="(b) Escala de Grises", use_container_width=True)
            col_blur_sup.image(blur_sup, caption="(c) Filtrado Gaussiano", use_container_width=True)
            
            # ROI Inferior
            st.markdown("**Región Inferior (Módulos en Suelo)**")
            col_rgb_inf, col_gray_inf, col_blur_inf = st.columns(3)
            col_rgb_inf.image(roi_inf, caption="(a) ROI Inferior RGB", use_container_width=True)
            col_gray_inf.image(gray_inf, caption="(b) Escala de Grises", use_container_width=True)
            col_blur_inf.image(blur_inf, caption="(c) Filtrado Gaussiano", use_container_width=True)
            
            st.divider()
            st.subheader("Análisis de Distribución de Amplitud (Histogramas)")
            fig_hist, axs_hist = plt.subplots(1, 2, figsize=(14, 4.5))
            
            axs_hist[0].hist(blur_sup.ravel(), bins=256, range=[0, 256], color='#1f77b4', alpha=0.7)
            axs_hist[0].axvline(x=threshold_sup, color='r', linestyle='--', linewidth=2.5)
            axs_hist[0].set_title("Histograma ROI Superior")
            
            axs_hist[1].hist(blur_inf.ravel(), bins=256, range=[0, 256], color='#2ca02c', alpha=0.7)
            axs_hist[1].axvline(x=threshold_inf, color='r', linestyle='--', linewidth=2.5)
            axs_hist[1].set_title("Histograma ROI Inferior")
            
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with tab2:
            st.subheader("Operaciones No Lineales de Amplitud")
            st.markdown("Estrategia implementada para contrarrestar los gradientes de iluminación en la zona superior:")
            
            col_adap, col_glob = st.columns(2)
            col_adap.image(mascara_sombra, caption=f"Umbral Adaptativo (Sombra) - Bloque: {block_size}, C: {c_value}")
            col_glob.image(mascara_sol, caption=f"Umbral Global Inverso (Luz) - Umbral: {threshold_sup}")
            
            st.image(mask_sup, caption="Máscara Superior: Final Combinado (Unión Espacial)", use_container_width=True)
            st.divider()
            st.image(mask_inf, caption=f"Máscara Inferior: Umbral Global Fijo - Umbral: {threshold_inf}", use_container_width=True)

        with tab3:
            st.subheader("Postprocesamiento Geométrico")
            st.markdown("Secuencia de limpieza para corregir la segmentación binaria (Cierre, Rellenado y Erosión).")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.image(mask_sup, caption="(a) Superior: Máscara Inicial (Cruda)", use_container_width=True)
                st.image(mask_sup_clean, caption="(c) Superior: Final Refinada (Post-Morfología)", use_container_width=True)
            with col_m2:
                st.image(mask_inf, caption="(a) Inferior: Máscara Inicial (Cruda)", use_container_width=True)
                st.image(mask_inf_clean, caption="(c) Inferior: Final Refinada (Post-Morfología)", use_container_width=True)

        with tab4:
            st.subheader("Detección de Componentes Conexas")
            
            roi_sup_draw = roi_sup.copy()
            roi_inf_draw = roi_inf.copy()
            cv2.drawContours(roi_sup_draw, paneles_sup, -1, (255, 0, 100), 3) # Usando magenta/rojo
            cv2.drawContours(roi_inf_draw, paneles_inf, -1, (255, 0, 100), 3)

            col_final_1, col_final_2 = st.columns(2)
            col_final_1.image(roi_sup_draw, caption=f"Superior - {conteo_sup} Paneles (Exactitud: {min(conteo_sup/conteo_real_sup, 1.0)*100:.1f}%)", use_container_width=True)
            col_final_2.image(roi_inf_draw, caption=f"Inferior - {conteo_inf} Paneles (Exactitud: {min(conteo_inf/conteo_real_inf, 1.0)*100:.1f}%)", use_container_width=True)
            
            st.divider()
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            col_met1.metric("Paneles Sup. Detectados", conteo_sup, f"Real: {conteo_real_sup}")
            col_met2.metric("Error Porcentual Sup.", f"{error_pct_sup:.2f}%")
            col_met3.metric("Paneles Inf. Detectados", conteo_inf, f"Real: {conteo_real_inf}")
            col_met4.metric("Error Porcentual Inf.", f"{error_pct_inf:.2f}%")

    except Exception as e:
        st.error(f"Error crítico durante la ejecución del pipeline: {e}")

st.divider()
st.caption("Desarrollado para el curso de Procesamiento de Señales I - Universidad del Magdalena - 2026")
