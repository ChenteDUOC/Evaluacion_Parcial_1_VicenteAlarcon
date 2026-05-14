"""
Módulo Pipeline para `ml_modeling`.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_classification, train_regression, train_clustering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_classification,
                inputs="dataset_final",
                outputs=["modelo_clasificacion", "reporte_clasificacion"],
                name="train_classification_node",
            ),
            node(
                func=train_regression,
                inputs="dataset_final",
                outputs=["modelo_regresion", "reporte_regresion"],
                name="train_regression_node",
            ),
            node(
                func=train_clustering,
                inputs="dataset_final",
                outputs=["modelo_clustering", "reporte_clustering"],
                name="train_clustering_node",
            ),
        ]
    )
