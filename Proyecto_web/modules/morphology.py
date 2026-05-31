import cv2
import numpy as np

def clean_mask(mask):
    """
    Aplica operaciones morfológicas para mejorar
    la máscara binaria.

    Etapas:
    1. Cierre morfológico.
    2. Rellenado de objetos.
    3. Erosión para separar paneles cercanos.

    Parámetros:
        mask: Máscara binaria.

    Retorna:
        Máscara procesada.
    """

    # Cierre morfológico:
    # rellena pequeños huecos y discontinuidades.
    kernel_cierre = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_cierre
    )

    # Obtención de contornos externos
    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Creación de una máscara vacía
    mask_llena = np.zeros_like(mask)

    # Rellenado completo de los objetos
    cv2.drawContours(
        mask_llena,
        contornos,
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Erosión para evitar que paneles adyacentes
    # queden unidos.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask_final = cv2.erode(
        mask_llena,
        kernel,
        iterations=3
    )

    return mask_final