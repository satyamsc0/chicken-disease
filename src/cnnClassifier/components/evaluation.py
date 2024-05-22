import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import mlflow
import mlflow.tensorflow




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        mlflow.set_experiment(self.config.experiment_name)  # Set or create an experiment in MLflow

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        # Start an MLflow run
        with mlflow.start_run(run_name=self.config.run_name):
            # Log model parameters
            mlflow.log_params({
                "batch_size": self.config.params_batch_size,
                "image_size": self.config.params_image_size,
                "model_path": str(self.config.path_of_model),
                "epochs": self.config.all_params.EPOCHS,
                "learning_rate": self.config.all_params.LEARNING_RATE,
                "number_of_classes": self.config.all_params.CLASSES
            })

            # Evaluate the model
            self.score = model.evaluate(self.valid_generator)
            
            # Log evaluation metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # Log the model
            mlflow.tensorflow.log_model(model, artifact_path="model")

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    

    