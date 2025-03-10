from typing import List
import mlflow
import pickle

class Doc2VecWraper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: List[str]) -> List[float]:
        return self.model.infer_vector(model_input)