import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from scripts.amg import write_masks_to_folder
# sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
# print(sum(p.numel() for p in sam.image_encoder.parameters()))
# print(sum(p.numel() for p in sam.prompt_encoder.parameters()))
# print(sum(p.numel() for p in sam.mask_decoder.parameters()))
sam.to(device="cuda")

mask_generator = SamAutomaticMaskGenerator(sam)
image = cv2.imread("notebooks/images/dog.jpg")
masks = mask_generator.generate(image)
write_masks_to_folder(masks,"output")
