from kedro.pipeline import Pipeline, node, pipeline
from .nodes import integrar_datos

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=integrar_datos,
            inputs=["ventas_cleaned", "productos_cleaned", "clientes_cleaned", "devoluciones_cleaned"],
            outputs="dataset_final",
            name="unir_datasets"
        )
    ])