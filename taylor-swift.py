from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import (
    InferenceConfiguration,
    VisualisationResponseFormat,
)
import cv2

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key=os.environ["API_KEY"]
)

client.configure(
    InferenceConfiguration(
        output_visualisation_format=VisualisationResponseFormat.NUMPY
    )
)

result = client.infer_from_workflow(
    workspace_name="capjamesg",
    workflow_name="taylor-swift",
    images={"image": "image.jpeg"},
)

classes = ["red", "reputation", "speak now", "midnights", "something else"]

output = result["output"]
predictions = result["predictions"]

for out, pred in zip(output, predictions):
    max_idx = pred.index(max(pred))
    if pred[max_idx] < 0.28:
        continue

    print(f"Class: {classes[max_idx]}")

    out_img = out["value"]
    cv2.imshow("output", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
