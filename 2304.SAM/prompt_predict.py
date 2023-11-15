import cv2
import numpy as np

from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)
image = cv2.imread("notebooks/images/truck.jpg")
predictor.set_image(image)

input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])  # ,[576, 751]
input_label = np.array([0])  # ,0
# input_point = np.array([[575, 750], [576, 751]])
# input_label = np.array([0, 0])

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=True,
)
print(masks)
cv2.imwrite("filename.png", masks[0] * 255)
