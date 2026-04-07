from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validar_calidad_datos

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validar_calidad_datos,
            inputs=["dataset_final", "ventas_raw"],
            outputs="reporte_validacion",
            name="nodo_validacion_final"
        )
    ])