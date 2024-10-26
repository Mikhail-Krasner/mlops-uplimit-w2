from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger

from src.data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier",
    version="0.1",
)

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, simple_model_handle: DeploymentHandle) -> None:
        # Define logger for this deployment
        self.logger = logger
        self.logger.add("api_ingress_{time}.log", rotation="1 day", retention="7 days")

        self.handle = simple_model_handle
        self.logger.info("APIIngress deployment initialized with simple model handle.")

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest):
        self.logger.info("Received prediction request: {}", request)
        try:
            # Use the handle to call the remote function for prediction
            result = await self.handle.predict.remote(request.review)
            self.logger.info("Prediction result received: {}", result)

            # Validate and return the response
            return SimpleModelResponse.model_validate(result.model_dump())
        except Exception as e:
            self.logger.error("Error during prediction: {}", e)
            return {"error": "An error occurred during prediction."}


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class SimpleModel:
    def __init__(self) -> None:
        # Define logger for this deployment
        self.logger = logger
        self.logger.add("simple_model_{time}.log", rotation="1 day", retention="7 days")

        self.logger.info("Loading the sentiment model...")
        self.session = Model.load_model()
        self.logger.info("Model loaded successfully.")

    def predict(self, review: str) -> SimpleModelResults:
        self.logger.info("Predicting sentiment for review: {}", review)
        try:
            # Use the Model's predict method to get the result
            result = self.session.predict(review)
            self.logger.info("Prediction successful: {}", result)

            # Validate and return the prediction results
            return SimpleModelResults.model_validate(result)
        except Exception as e:
            self.logger.error("Error during model prediction: {}", e)
            raise


# Bind the entry point to SimpleModel
entrypoint = APIIngress.bind(
    SimpleModel.bind(),
)
