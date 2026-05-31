import cv2
import numpy as np

def segment_upper(blur_sup):
    """
    Segmenta el ROI superior.

    Debido a la presencia de una sombra,
    se utiliza una estrategia híbrida:

    - Umbral adaptativo en la zona sombreada.
    - Umbral global en la zona iluminada.

    Parámetros:
        blur_sup: ROI superior suavizada.

    Retorna:
        Máscara binaria.
    """

    # Posición aproximada donde termina la sombra
    corte = 510

    sombra = blur_sup[:, :corte]
    sol = blur_sup[:, corte:]

    # Segmentación de la zona iluminada
    _, mascara_sol = cv2.threshold(
        sol,
        81,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Segmentación de la zona sombreada
    mascara_sombra = cv2.adaptiveThreshold(
        sombra,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12
    )

    # Reconstrucción de la máscara completa
    mascara = np.hstack(
        (mascara_sombra, mascara_sol)
    )

    return mascara


def segment_lower(blur_inf):
    """
    Segmenta el ROI inferior mediante
    umbralización global.

    Parámetros:
        blur_inf: ROI inferior suavizada.

    Retorna:
        Máscara binaria.
    """

    _, mascara = cv2.threshold(
        blur_inf,
        95,
        255,
        cv2.THRESH_BINARY_INV
    )

    return mascara