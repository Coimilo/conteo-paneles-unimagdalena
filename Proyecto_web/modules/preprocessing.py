import cv2

def extract_rois(img_rgb):
    """
    Extrae las regiones de interés (ROI) que contienen
    exclusivamente los paneles solares.

    El objetivo es reducir la cantidad de información
    irrelevante antes del procesamiento.

    Parámetros:
        img_rgb: Imagen original RGB.

    Retorna:
        roi_sup, roi_inf
    """

    roi_sup = img_rgb[80:445, 1120:2925]
    roi_inf = img_rgb[1634:2154, 940:3140]

    return roi_sup, roi_inf


def grayscale_and_blur(roi):
    """
    Convierte la ROI a escala de grises y aplica
    un filtro Gaussiano para reducir ruido.

    Parámetros:
        roi: Región de interés RGB.

    Retorna:
        gray: Imagen en escala de grises.
        blur: Imagen suavizada.
    """

    # Conversión RGB -> Gris
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_RGB2GRAY
    )

    # Suavizado espacial
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray, blur