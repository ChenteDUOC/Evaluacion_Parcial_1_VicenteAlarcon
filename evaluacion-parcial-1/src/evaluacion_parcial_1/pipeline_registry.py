from typing import Dict
from kedro.pipeline import Pipeline
from evaluacion_parcial_1.pipelines import data_ingestion, data_cleaning, data_integration, data_validation, ml_modeling

def register_pipelines() -> Dict[str, Pipeline]:
    ingestion_pipeline = data_ingestion.create_pipeline()
    cleaning_pipeline = data_cleaning.create_pipeline()
    integration_pipeline = data_integration.create_pipeline()
    validation_pipeline = data_validation.create_pipeline()

    modeling_pipeline = ml_modeling.create_pipeline()

    return {
        "ingestion": ingestion_pipeline,
        "cleaning": cleaning_pipeline,
        "integration": integration_pipeline,
        "validation": validation_pipeline,
        "modeling": modeling_pipeline,
        "__default__": ingestion_pipeline + cleaning_pipeline + integration_pipeline + validation_pipeline + modeling_pipeline,
    }