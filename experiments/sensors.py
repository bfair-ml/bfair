from bfair.datasets import load_review
from bfair.datasets.reviews import REVIEW_COLUMN
from bfair.sensors import SensorHandler, EmbeddingBasedSensor
from autogoal.kb import Text

dataset = load_review(split_seed=0)
sensor = EmbeddingBasedSensor.build_default_in_hierarchy_mode(
    language="english", source="word2vec-debiased"
)
handler = SensorHandler(sensors=[sensor])
reviews = dataset.data[REVIEW_COLUMN]
for text in reviews:
    annotations = handler.annotate(text, Text, ["male", "female"])
    print(annotations)
