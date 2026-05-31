def evaluate(real, estimated):
    """
    Calcula error absoluto y porcentual.
    """

    error_abs = abs(real - estimated)

    error_pct = (
        error_abs / real * 100
        if real > 0
        else 0
    )

    return error_abs, error_pct


def print_evaluation(
    real_sup,
    est_sup,
    real_inf,
    est_inf
):
    """
    Imprime las métricas de evaluación.
    """

    error_abs_sup, error_pct_sup = evaluate(
        real_sup,
        est_sup
    )

    error_abs_inf, error_pct_inf = evaluate(
        real_inf,
        est_inf
    )

    print("\n========== RESULTADOS ==========")
    print(f"Paneles ROI Superior: {est_sup}")
    print(f"Paneles ROI Inferior: {est_inf}")

    print("\n========== MÉTRICAS DE EVALUACIÓN ==========")

    print(
        f"ROI Superior -> Real: {real_sup}, Estimado: {est_sup}"
    )
    print(
        f"Error Absoluto Superior: {error_abs_sup}"
    )
    print(
        f"Error Porcentual Superior: {error_pct_sup:.2f}%"
    )

    print("-" * 40)

    print(
        f"ROI Inferior -> Real: {real_inf}, Estimado: {est_inf}"
    )
    print(
        f"Error Absoluto Inferior: {error_abs_inf}"
    )
    print(
        f"Error Porcentual Inferior: {error_pct_inf:.2f}%"
    )