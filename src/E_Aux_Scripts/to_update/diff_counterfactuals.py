import cv2
import numpy as np
import os

counterfactual_dir = "../../data/images/repaint_results"
images_base_dir = "../../data/images/VinDr-Mammo-Clip-CLAHE-512"
masks_dir = "../../data/images/masks_512"

mse_total = 0
count = 0

for filename in os.listdir(counterfactual_dir):
    img_path = os.path.join(images_base_dir, filename)
    cf_path = os.path.join(counterfactual_dir, filename)
    mask_path = os.path.join(masks_dir, filename)

    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        raise FileNotFoundError(f"Missing file for {filename}")

    img = cv2.imread(img_path)
    cf = cv2.imread(cf_path)
    mask = cv2.imread(mask_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cf = cv2.cvtColor(cf, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32)
    binary_mask = mask / 255.0

    # Apply mask
    img_masked = img * binary_mask
    cf_masked = cf * binary_mask

    def mse(img1, img2):
        h, w = img1.shape
        diff = img1 - img2
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse

    mse_value = mse(img_masked, cf_masked)

    mse_total += mse_value
    count += 1

if count > 0:
    print("Mean MSE over all images:", mse_total / count)
else:
    print("No images processed.")