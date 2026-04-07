from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generar_diagnostico

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generar_diagnostico,
            inputs=["ventas_raw", "productos_raw", "clientes_raw", "devoluciones_raw"],
            outputs="reporte_diagnostico",
            name="nodo_diagnostico_inicial"
        )
    ])