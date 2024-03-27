import os

import cv2
import supervision as sv
from inference_sdk import (InferenceConfiguration, InferenceHTTPClient,
                           VisualisationResponseFormat)

SPECIFICATION = {
    "specification": {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "InstanceSegmentationModel",
                "name": "coffee_detection",
                "image": "$inputs.image",
                "model_id": "coffee-bean-detection-kasdb/2",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.coffee_detection.*",
            }
        ],
    }
}

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"]
)

client.configure(
    InferenceConfiguration(
        output_visualisation_format=VisualisationResponseFormat.NUMPY
    )
)

image = cv2.imread("./coffee.jpg")

result = client.infer_from_workflow(
    specification=SPECIFICATION["specification"],
    images={"image": image},
)

detections = sv.Detections.from_inference(result["predictions"][0])

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)

sv.plot_image(annotated_frame)
