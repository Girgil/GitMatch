import os
import mlflow
import mlflow.pyfunc
import pickle
import os
import sys
from mlflow.tracking import MlflowClient

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.Doc2VecWraper import Doc2VecWraper
#import Doc2VecWraper

class MLFlowManager:
    def __init__(self, staging_alias="staging", production_alias="production", archive_alias="archived"):
        self.client = MlflowClient()
        self.staging_alias = staging_alias
        self.production_alias = production_alias
        self.archive_alias = archive_alias

    def register_model(self, model, model_name, artifact_path, stage="staging"):
        """
        Enregistrement du nouveau modèle
        """
        with mlflow.start_run() as run:
            # Enregistrement du modèle
            model_path = f"{artifact_path}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            Doc2VecWraper_path = os.getcwd()
            #Doc2VecWraper_path = os.path.join(current_directory, "model")
            
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=Doc2VecWraper(),
                artifacts={"model_path": model_path},
                code_paths=[Doc2VecWraper_path]
            )

            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            mlflow.register_model(model_uri, model_name)

            # Récupérer la dernière version
            latest_version = self.get_latest_version(model_name)

            # Ajouter tag et alias
            self.update_model_stage(model_name, latest_version.version, stage)

    def get_latest_version(self, model_name):
        versions = self.client.search_model_versions(f"name='{model_name}'")
        sorted_versions = sorted(versions, key=lambda x: x.creation_timestamp, reverse=True)
        return sorted_versions[0]

    def update_model_stage(self, model_name, version, stage):
        # Supprimer les anciens tags et aliases
        self.client.delete_model_version_tag(name=model_name, version=version, key="stage")
        self.client.delete_registered_model_alias(name=model_name, alias=stage)

        # Ajouter le nouveau tag et alias
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="stage",
            value=stage,
        )
        self.client.set_registered_model_alias(
            name=model_name,
            alias=stage,
            version=version
        )

    def promote_to_production(self, model_name):
        staging_version = self.client.get_model_version_by_alias(model_name, self.staging_alias)
        
        self.update_model_stage(model_name, staging_version.version, self.production_alias)

    def archive_model_from_production(self, model_name):
        production_version = self.client.get_model_version_by_alias(model_name, self.production_alias)

        self.update_model_stage(model_name, production_version.version, self.archive_alias)

    def archive_model_from_staging(self, model_name):
        staging_version = self.client.get_model_version_by_alias(model_name, self.staging_alias)

        self.update_model_stage(model_name, staging_version.version, self.archive_alias)

