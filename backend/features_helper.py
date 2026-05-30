import cv2
import numpy as np

def extract_features_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (256, 256))

    edges = cv2.Canny(img, 100, 200)

    hist = cv2.calcHist([img], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    feature_vector = np.hstack((edges.flatten()[:5000], hist))
    return feature_vector.reshape(1, -1)
