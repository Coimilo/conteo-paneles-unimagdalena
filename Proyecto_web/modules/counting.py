import cv2

def count_panels(mask, area_minima=4000):
    """
    Detecta y cuenta paneles solares a partir
    de una máscara binaria.

    Parámetros:
        mask: Máscara binaria.
        area_minima: Área mínima permitida.

    Retorna:
        Lista de contornos válidos.
    """

    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Eliminación de objetos pequeños
    # considerados ruido.
    contornos_validos = [
        c
        for c in contornos
        if cv2.contourArea(c) > area_minima
    ]

    return contornos_validos