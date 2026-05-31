import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

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
    layout="wide" # Usa todo el ancho de la pantalla
)

# =========================================================
# TÍTULO Y DESCRIPCIÓN (Basado en tu guía)
# =========================================================
st.title("☀️ Proyecto de Segmentación de Imágenes de Dron")
st.markdown("""
Esta aplicación web presenta los resultados del proyecto de **Procesamiento de Señales I** de la Universidad del Magdalena.
El objetivo es aplicar conceptos de procesamiento digital de señales para el **conteo de paneles solares** sobre el edificio docente, utilizando técnicas clásicas de visión artificial.
""")

# =========================================================
# JUSTIFICACIÓN TÉCNICA (Teoría de Señales)
# =========================================================
with st.expander("📘 Fundamentos de Señales Espaciales (Justificación Técnica)"):
    st.markdown("""
    **Marco Teórico - Procesamiento de Señales I**
    
    Para este análisis sobre el edificio docente de la Universidad del Magdalena, la imagen se aborda formalmente como una **señal discreta bidimensional**, donde cada píxel es una muestra espacial.
    
    *   **Filtrado Espacial (Pasa-bajas):** Antes de la segmentación, se aplica un suavizado Gaussiano. En el dominio de las frecuencias, esto actúa como un filtro pasa-bajas que atenúa el ruido de alta frecuencia (variaciones bruscas), estabilizando la señal.
    *   **Umbralización (Operación no lineal):** La segmentación es una operación no lineal sobre la amplitud de la señal espacial. Se utilizó una estrategia híbrida (adaptativa y global) para contrarrestar los cambios de iluminación y sombras.
    *   **Morfología Matemática:** Funciona como un filtro espacial no lineal de post-procesamiento para rellenar discontinuidades y separar frecuencias espaciales adyacentes (paneles muy juntos).
    
    *Desarrollado por: Camilo Cantillo, Luis Mercado*
    """)

# =========================================================
# BARRA LATERAL (SIDEBAR) - Panel de Control Interactivo
# =========================================================
# ¡Esto añade interactividad, que es un PLUS en tu rúbrica!
st.sidebar.header("⚙️ Panel de Control")

st.sidebar.subheader("1. Carga de Imagen")
# Definimos la ruta por defecto
default_image_path = os.path.join("images", "DJI_0613.JPG")

# Verificamos si la imagen existe para evitar errores
if not os.path.exists(default_image_path):
    st.sidebar.error(f"No se encontró la imagen en {default_image_path}. Por favor, verifica la carpeta 'images'.")
    st.stop()

# Botón para iniciar el procesamiento
run_process = st.sidebar.button("Procesar Imagen")

st.sidebar.divider()

# Sección de parámetros dinámicos (Plus de la guía: análisis de sensibilidad)
st.sidebar.subheader("2. Ajuste de Parámetros")
st.sidebar.write("Ajusta los umbrales para ver cómo afectan la segmentación en tiempo real.")

# Slider interactivo para el umbral superior
threshold_sup = st.sidebar.slider(
    "Umbral ROI Superior (Sol)",
    min_value=0,
    max_value=255,
    value=81, # Valor por defecto de tu código
    help="Ajusta el umbral para la zona iluminada de la ROI superior."
)

# Slider interactivo para el umbral inferior
threshold_inf = st.sidebar.slider(
    "Umbral ROI Inferior",
    min_value=0,
    max_value=255,
    value=95, # Valor por defecto de tu código
    help="Ajusta el umbral global para la ROI inferior."
)

# Input numérico para el área mínima
area_min = st.sidebar.number_input(
    "Área Mínima de Panel (px)",
    min_value=100,
    max_value=10000,
    value=4000, # Valor por defecto
    step=100
)

st.sidebar.divider()
st.sidebar.subheader("3. Filtro Adaptativo (Zona de Sombra)")
st.sidebar.write("Ajusta el comportamiento en la zona superior izquierda.")

# Slider para el Block Size (Asegurándonos de que el paso sea 2 para mantenerlo impar)
block_size = st.sidebar.slider(
    "Tamaño de Bloque (Block Size)",
    min_value=3,
    max_value=99,
    value=35,  # Tu valor original
    step=2,
    help="Define el área de vecindad espacial. Debe ser estrictamente un número impar."
)

# Slider para la Constante C
c_value = st.sidebar.slider(
    "Constante (C)",
    min_value=-20,
    max_value=50,
    value=12,  # Tu valor original
    step=1,
    help="Valor que se resta a la media local. Ajusta la sensibilidad del filtro frente a reflejos."
)

# Valores reales fijos para la evaluación
conteo_real_sup = 36
conteo_real_inf = 36


# =========================================================
# CUERPO PRINCIPAL DE LA PÁGINA
# =========================================================

# Sección que se ejecuta solo al pulsar el botón "Procesar Imagen"
if run_process:
    # Mostramos un spinner de carga
    with st.spinner('Procesando señal espacial (imagen)... esto puede tardar unos segundos.'):

        try:
            # 1. CARGA DE IMAGEN
            # ----------------------------------
            img_rgb = load_image(default_image_path)

            # 2. PREPROCESAMIENTO
            # ----------------------------------
            roi_sup, roi_inf = extract_rois(img_rgb)
            gray_sup, blur_sup = grayscale_and_blur(roi_sup)
            gray_inf, blur_inf = grayscale_and_blur(roi_inf)

            # 3. SEGMENTACIÓN (Usando los valores de los sliders laterales)
            # ----------------------------------
            # Recreamos la lógica de tus funciones pero usando los umbrales dinámicos
            
            # Segmentación Superior Híbrida
            
            # 3. SEGMENTACIÓN (Usando los valores de los sliders laterales)
            # ----------------------------------
            
            # Segmentación Superior Híbrida
            corte = 510
            sombra = blur_sup[:, :corte]
            sol = blur_sup[:, corte:]
            
            # Zona iluminada (Umbral global)
            _, mascara_sol = cv2.threshold(sol, threshold_sup, 255, cv2.THRESH_BINARY_INV)
            
            # Zona sombreada (Umbral adaptativo modificado)
            mascara_sombra = cv2.adaptiveThreshold(
                sombra, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                block_size, # <- Variable del slider
                c_value     # <- Variable del slider
            )
            
            mask_sup = np.hstack((mascara_sombra, mascara_sol))

            # Segmentación Inferior Global
            _, mask_inf = cv2.threshold(blur_inf, threshold_inf, 255, cv2.THRESH_BINARY_INV)

            # 4. POSTPROCESAMIENTO MORFOLÓGICO
            # ----------------------------------
            mask_sup_clean = clean_mask(mask_sup)
            mask_inf_clean = clean_mask(mask_inf)

            # 5. CONTEO DE PANELES (Usando el área mínima dinâmica)
            # ----------------------------------
            paneles_sup = count_panels(mask_sup_clean, area_minima=area_min)
            paneles_inf = count_panels(mask_inf_clean, area_minima=area_min)

            conteo_sup = len(paneles_sup)
            conteo_inf = len(paneles_inf)

            # 6. VALIDACIÓN CUANTITATIVA
            # ----------------------------------
            error_abs_sup, error_pct_sup = evaluate(conteo_real_sup, conteo_sup)
            error_abs_inf, error_pct_inf = evaluate(conteo_real_inf, conteo_inf)


            # =========================================================
            # VISUALIZACIÓN DE RESULTADOS EN LA GUI
            # =========================================================
            
            # --- SECCIÓN 1: IMAGEN ORIGINAL ---
            st.header("1. Imagen Original")
            st.image(img_rgb, caption="Vista aérea del edificio docente (RGB)", use_container_width=True)

            # --- SECCIÓN 2: MÉTRICAS DE EVALUACIÓN ---
            # La guía pide reportar métricas cuantitativas claras.
            st.header("2. Resultados del Conteo y Evaluación Cuantitativa")
            
            # Creamos dos columnas principales para Superior e Inferior
            col_met_sup, col_met_inf = st.columns(2)

            with col_met_sup:
                st.subheader("Región Superior")
                # Usamos componentes métricos de Streamlit para visualización profesional
                m1, m2, m3 = st.columns(3)
                m1.metric("Detectados", conteo_sup, help=f"Real: {conteo_real_sup}")
                m2.metric("Error Absoluto", error_abs_sup)
                m3.metric("Error Porcentual", f"{error_pct_sup:.2f}%")
                
                # Barra de progreso visual (Plus de UI)
                precision_sup = min(conteo_sup / conteo_real_sup, 1.0) if conteo_real_sup > 0 else 0
                st.progress(precision_sup, text=f"Precisión visual del conteo: {precision_sup*100:.1f}%")

            with col_met_inf:
                st.subheader("Región Inferior")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detectados", conteo_inf, help=f"Real: {conteo_real_inf}")
                m2.metric("Error Absoluto", error_abs_inf)
                m3.metric("Error Porcentual", f"{error_pct_inf:.2f}%")

                # Barra de progreso visual
                precision_inf = min(conteo_inf / conteo_real_inf, 1.0) if conteo_real_inf > 0 else 0
                st.progress(precision_inf, text=f"Precisión visual del conteo: {precision_inf*100:.1f}%")

            st.divider()

            # --- SECCIÓN 3: VISUALIZACIÓN DEL PROCESO (Flujo de Señales) ---
            # La guía pide mostrar evidencias visuales de cada etapa.
            # Usamos pestañas (tabs) para organizar el flujo (Plus de UI).
            st.header("3. Visualización del Flujo de Procesamiento de Señales")
            
            tab_roi, tab_mask, tab_final = st.tabs([
                "1. Regiones de Interés (ROIs)",
                "2. Máscaras de Segmentación",
                "3. Detección y Conteo Final"
            ])

            with tab_roi:
                st.subheader("Extracción y Preprocesamiento de ROIs")
                col_gray_sup, col_gray_inf = st.columns(2)
                col_gray_sup.image(gray_sup, caption="ROI Superior - Escala de Grises", clamp=True)
                col_gray_inf.image(gray_inf, caption="ROI Inferior - Escala de Grises", clamp=True)
                st.markdown("**Nota:** Las señales espaciales se han convertido a escala de grises y suavizado con un filtro Gaussiano (5x5).")

            with tab_mask:
                st.subheader("Máscaras Binarias Limpias (Post-Morfología)")
                col_mask_sup, col_mask_inf = st.columns(2)
                # Streamlit muestra arrays de numpy como imágenes, clamp asegura el rango 0-255
                col_mask_sup.image(mask_sup_clean, caption=f"Máscara Superior (Umbral Sol: {threshold_sup})", clamp=True)
                col_mask_inf.image(mask_inf_clean, caption=f"Máscara Inferior (Umbral: {threshold_inf})", clamp=True)
                st.markdown("**Proceso:** Segmentación (adaptativa/global) -> Operaciones Morfológicas (cierre, rellenado, 3 erosiones).")

            with tab_final:
                st.subheader("Detección de Contornos sobre la Imagen Original")
                
                # Crear copias para dibujar contornos y mostrar en la GUI
                roi_sup_draw = roi_sup.copy()
                roi_inf_draw = roi_inf.copy()
                cv2.drawContours(roi_sup_draw, paneles_sup, -1, (255, 0, 0), 3) # Contornos azules, grosor 3
                cv2.drawContours(roi_inf_draw, paneles_inf, -1, (255, 0, 0), 3)

                col_draw_sup, col_draw_inf = st.columns(2)
                col_draw_sup.image(roi_sup_draw, caption=f"Detección Superior: {conteo_sup} paneles")
                col_draw_inf.image(roi_inf_draw, caption=f"Detección Inferior: {conteo_inf} paneles")
                st.markdown(f"**Criterio:** Se detectan contornos externos válidos con un área mayor a **{area_min} píxeles**.")

        except Exception as e:
            st.error(f"Ocurrió un error inesperado durante el procesamiento: {e}")

else:
    # Mensaje inicial cuando la página carga por primera vez
    st.info("👈 Ajusta los parámetros en el panel de control lateral y haz clic en 'Procesar Imagen' para ver el análisis.")
    
    # Mostrar una vista previa de la imagen original
    img_preview = Image.open(default_image_path)
    st.image(img_preview, caption="Vista previa de la imagen de entrada", use_container_width=True)

# =========================================================
# PIE DE PÁGINA
# =========================================================
st.divider()
st.caption("Desarrollado para el curso de Procesamiento de Señales I - Universidad del Magdalena - 2026")