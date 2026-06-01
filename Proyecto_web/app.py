import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Importar tus módulos personalizados (desde la carpeta 'modules')
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

# =========================================================
# TÍTULO Y DESCRIPCIÓN 
# =========================================================
st.title("☀️ Proyecto de Segmentación de Imágenes de Dron")
st.markdown("""
Esta aplicación web presenta los resultados del proyecto de **Procesamiento de Señales I** de la Universidad del Magdalena.
El objetivo es aplicar conceptos de procesamiento digital de señales para el **conteo de paneles solares** sobre el edificio docente, utilizando técnicas clásicas de visión artificial.
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
# BARRA LATERAL (SIDEBAR) - Panel de Control Interactivo
# =========================================================
st.sidebar.header("⚙️ Panel de Control")

st.sidebar.subheader("1. Selección de Imagen")

# Menú simplificado únicamente con las dos imágenes optimizadas
opcion_imagen = st.sidebar.selectbox(
    "Seleccione la imagen de análisis:",
    ["DJI_0612.JPG (Predeterminada)", "DJI_0613.JPG"]
)

# Determinación de la ruta del archivo en disco
# =========================================================
# SOLUCIÓN DE RUTA DINÁMICA PARA LA NUBE
# =========================================================
# 1. Obtenemos la ruta absoluta de la carpeta donde vive este app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Definimos el nombre exacto del archivo (¡Ojo con las mayúsculas/minúsculas!)
nombre_archivo = "DJI_0613.JPG" if "0613" in opcion_imagen else "DJI_0612.JPG"

# 3. Construimos la ruta uniendo la carpeta del script + 'images' + el archivo
image_path = os.path.join(BASE_DIR, "images", nombre_archivo)

# Validación de existencia
img_ready = os.path.exists(image_path)

if not img_ready:
    st.sidebar.error(f"No se encontró '{nombre_archivo}' en la ruta calculada: {image_path}")
    st.sidebar.warning("👉 Verifica en GitHub si la extensión está en minúsculas (.jpg) o mayúsculas (.JPG)")

st.sidebar.divider()

# Botón para ejecutar el flujo de procesamiento
run_process = st.sidebar.button("Procesar Imagen", disabled=not img_ready)

st.sidebar.divider()

st.sidebar.subheader("2. Ajuste de Umbrales Globales")
threshold_sup = st.sidebar.slider("Umbral ROI Superior (Sol)", min_value=0, max_value=255, value=81)
threshold_inf = st.sidebar.slider("Umbral ROI Inferior", min_value=0, max_value=255, value=90)
area_min = st.sidebar.number_input("Área Mínima de Panel (px)", min_value=100, max_value=10000, value=4000, step=100)

st.sidebar.divider()

st.sidebar.subheader("3. Filtro Adaptativo (Zona de Sombra)")
block_size = st.sidebar.slider("Tamaño de Bloque (Block Size)", min_value=3, max_value=99, value=35, step=2)
c_value = st.sidebar.slider("Constante (C)", min_value=-20, max_value=50, value=12, step=1)

# Valores reales fijos según el plano de referencia para la evaluación
conteo_real_sup = 36
conteo_real_inf = 36


# =========================================================
# CUERPO PRINCIPAL DE LA PÁGINA
# =========================================================

if run_process and img_ready:
    with st.spinner('Procesando señal espacial...'):
        try:
            # 1. CARGA DE IMAGEN DESDE DISCO
            # ----------------------------------
            img_rgb = load_image(image_path)

            # 2. PREPROCESAMIENTO
            # ----------------------------------
            roi_sup, roi_inf = extract_rois(img_rgb)
            gray_sup, blur_sup = grayscale_and_blur(roi_sup)
            gray_inf, blur_inf = grayscale_and_blur(roi_inf)

            # 3. SEGMENTACIÓN PARAMÉTRICA DINÁMICA
            # ----------------------------------
            corte = 510
            sombra = blur_sup[:, :corte]
            sol = blur_sup[:, corte:]
            
            # Umbralización global adaptada al control deslizante
            _, mascara_sol = cv2.threshold(sol, threshold_sup, 255, cv2.THRESH_BINARY_INV)
            
            # Umbralización local/adaptativa vinculada a los controles deslizantes
            mascara_sombra = cv2.adaptiveThreshold(
                sombra, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, c_value
            )
            mask_sup = np.hstack((mascara_sombra, mascara_sol))

            _, mask_inf = cv2.threshold(blur_inf, threshold_inf, 255, cv2.THRESH_BINARY_INV)

            # 4. POSTPROCESAMIENTO MORFOLÓGICO
            # ----------------------------------
            mask_sup_clean = clean_mask(mask_sup)
            mask_inf_clean = clean_mask(mask_inf)

            # 5. CONTEO DE PANELES BASADO EN CONTOURS
            # ----------------------------------
            paneles_sup = count_panels(mask_sup_clean, area_minima=area_min)
            paneles_inf = count_panels(mask_inf_clean, area_minima=area_min)

            conteo_sup = len(paneles_sup)
            conteo_inf = len(paneles_inf)

            # 6. VALIDACIÓN ESTADÍSTICA CUANTITATIVA
            # ----------------------------------
            error_abs_sup, error_pct_sup = evaluate(conteo_real_sup, conteo_sup)
            error_abs_inf, error_pct_inf = evaluate(conteo_real_inf, conteo_inf)

            # === RENDERIZADO DE RESULTADOS EN LA INTERFAZ ===
            
            # --- SECCIÓN 1: IMAGEN ORIGINAL ---
            st.header("1. Imagen Seleccionada")
            st.image(img_rgb, caption=f"Análisis espacial sobre el archivo: {nombre_archivo}", use_container_width=True)

            # --- SECCIÓN 2: MÉTRICAS DE EVALUACIÓN ---
            st.header("2. Resultados del Conteo y Evaluación Cuantitativa")
            col_met_sup, col_met_inf = st.columns(2)

            with col_met_sup:
                st.subheader("Región Superior (Techo Edificio Docente)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detectados", conteo_sup, help=f"Valor real de referencia: {conteo_real_sup}")
                m2.metric("Error Absoluto", error_abs_sup)
                m3.metric("Error Porcentual", f"{error_pct_sup:.2f}%")
                precision_sup = min(conteo_sup / conteo_real_sup, 1.0) if conteo_real_sup > 0 else 0
                st.progress(precision_sup, text=f"Exactitud del algoritmo: {precision_sup*100:.1f}%")

            with col_met_inf:
                st.subheader("Región Inferior (Módulos en Suelo)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detectados", conteo_inf, help=f"Valor real de referencia: {conteo_real_inf}")
                m2.metric("Error Absoluto", error_abs_inf)
                m3.metric("Error Porcentual", f"{error_pct_inf:.2f}%")
                precision_inf = min(conteo_inf / conteo_real_inf, 1.0) if conteo_real_inf > 0 else 0
                st.progress(precision_inf, text=f"Exactitud del algoritmo: {precision_inf*100:.1f}%")

            st.divider()

            # --- SECCIÓN 3: PESTAÑAS INTERACTIVAS ---
            st.header("3. Visualización del Flujo de Procesamiento de Señales")
            tab_roi, tab_mask, tab_final = st.tabs([
                "1. Regiones de Interés (ROIs) e Histogramas",
                "2. Máscaras de Segmentación",
                "3. Detección y Conteo Final"
            ])

            with tab_roi:
                st.subheader("Extracción y Preprocesamiento de ROIs")
                col_gray_sup, col_gray_inf = st.columns(2)
                col_gray_sup.image(gray_sup, caption="ROI Superior - Escala de Grises", use_container_width=True)
                col_gray_inf.image(gray_inf, caption="ROI Inferior - Escala de Grises", use_container_width=True)
                
                st.divider()
                
                st.subheader("📊 Análisis de Distribución de Amplitud (Histogramas)")
                st.markdown("La línea discontinua roja representa el **umbral de corte** seleccionado actualmente en el panel lateral.")
                
                # Renderizado de los dos Histogramas de Frecuencia Espacial
                fig_hist, axs_hist = plt.subplots(1, 2, figsize=(14, 4.5))
                
                # Gráfico ROI Superior
                axs_hist[0].hist(blur_sup.ravel(), bins=256, range=[0, 256], color='#1f77b4', alpha=0.7, rwidth=0.9)
                axs_hist[0].axvline(x=threshold_sup, color='r', linestyle='--', linewidth=2.5, label=f"Umbral = {threshold_sup}")
                axs_hist[0].set_title("Histograma Frecuencias Espaciales - ROI Superior")
                axs_hist[0].set_xlabel("Intensidad de Gris")
                axs_hist[0].set_ylabel("Píxeles")
                axs_hist[0].legend(loc="upper right")
                axs_hist[0].grid(True, alpha=0.3)
                
                # Gráfico ROI Inferior
                axs_hist[1].hist(blur_inf.ravel(), bins=256, range=[0, 256], color='#2ca02c', alpha=0.7, rwidth=0.9)
                axs_hist[1].axvline(x=threshold_inf, color='r', linestyle='--', linewidth=2.5, label=f"Umbral = {threshold_inf}")
                axs_hist[1].set_title("Histograma Frecuencias Espaciales - ROI Inferior")
                axs_hist[1].set_xlabel("Intensidad de Gris")
                axs_hist[1].set_ylabel("Píxeles")
                axs_hist[1].legend(loc="upper right")
                axs_hist[1].grid(True, alpha=0.3)
                
                fig_hist.tight_layout()
                st.pyplot(fig_hist)
                plt.close(fig_hist)  # Liberación de memoria del objeto gráfico

            with tab_mask:
                st.subheader("Máscaras Binarias Limpias (Post-Morfología)")
                col_mask_sup, col_mask_inf = st.columns(2)
                col_mask_sup.image(mask_sup_clean, caption=f"Máscara Superior (Block Size: {block_size}, C: {c_value})", use_container_width=True)
                col_mask_inf.image(mask_inf_clean, caption=f"Máscara Inferior (Umbral Global: {threshold_inf})", use_container_width=True)

            with tab_final:
                st.subheader("Detección de Contornos sobre la Imagen Original")
                roi_sup_draw = roi_sup.copy()
                roi_inf_draw = roi_inf.copy()
                cv2.drawContours(roi_sup_draw, paneles_sup, -1, (255, 0, 0), 3)
                cv2.drawContours(roi_inf_draw, paneles_inf, -1, (255, 0, 0), 3)

                col_draw_sup, col_draw_inf = st.columns(2)
                col_draw_sup.image(roi_sup_draw, caption=f"Detección Superior: {conteo_sup} de {conteo_real_sup} paneles", use_container_width=True)
                col_draw_inf.image(roi_inf_draw, caption=f"Detección Inferior: {conteo_inf} de {conteo_real_inf} paneles", use_container_width=True)

        except Exception as e:
            st.error(f"Error crítico durante la ejecución del pipeline: {e}")

else:
    st.info("👈 Seleccione el dataset que desea analizar en la barra lateral y haga clic en 'Procesar Imagen'.")
    
    # Vista previa estática antes de ejecutar el algoritmo de segmentación
    if img_ready:
        st.image(Image.open(image_path), caption=f"Vista previa del dataset seleccionado: {nombre_archivo}", use_container_width=True)

# =========================================================
# PIE DE PÁGINA
# =========================================================
st.divider()
st.caption("Desarrollado para el curso de Procesamiento de Señales I - Universidad del Magdalena - 2026")
