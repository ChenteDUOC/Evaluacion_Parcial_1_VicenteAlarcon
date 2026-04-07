from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_ventas, preprocess_productos, preprocess_clientes, preprocess_devoluciones

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_ventas, 
            inputs="ventas_raw", 
            outputs="ventas_cleaned", 
            name="limpiar_ventas"
        ),
        node(
            func=preprocess_productos, 
            inputs="productos_raw", 
            outputs="productos_cleaned", 
            name="limpiar_productos"
        ),
        node(
            func=preprocess_clientes, 
            inputs="clientes_raw", 
            outputs="clientes_cleaned", 
            name="limpiar_clientes"
        ),
        node(
            func=preprocess_devoluciones, 
            inputs="devoluciones_raw", 
            outputs="devoluciones_cleaned", 
            name="limpiar_devoluciones"
        ),
    ])