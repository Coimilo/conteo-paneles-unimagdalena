import cv2

def load_image(path):
    """
    Carga una imagen desde disco y la convierte de BGR a RGB.

    Parámetros:
        path (str): Ruta de la imagen.

    Retorna:
        numpy.ndarray: Imagen en formato RGB.
    """

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen: {path}"
        )

    # OpenCV carga imágenes en formato BGR.
    # Se convierte a RGB para visualización con Matplotlib.
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb