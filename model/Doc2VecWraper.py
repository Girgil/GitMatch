from typing import List
import mlflow
import pickle

class Doc2VecWraper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Permet le chargement du modèle
        """
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: List[str]) -> List[float]:
        """
        Permet la prédiction
        """
        return self.model.infer_vector(model_input)