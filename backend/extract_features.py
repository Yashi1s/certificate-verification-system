import cv2
import numpy as np
import pytesseract
from skimage.feature import hog
from pdf2image import convert_from_path
import os
import tempfile
import uuid


# ---------------------------------------
# Image Quality Metrics (Blur + Noise)
# ---------------------------------------
def get_image_quality_metrics(gray):

    # Blur detection using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Noise estimation using pixel standard deviation
    noise_score = np.std(gray)

    return blur_score, noise_score


# ---------------------------------------
# Load Image (PDF or Image)
# ---------------------------------------
def load_image(path):

    ext = os.path.splitext(path)[1].lower()

    # ---------- PDF HANDLING ----------
    if ext == ".pdf":

        images = convert_from_path(path, dpi=300)

        temp_name = f"pdf_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(tempfile.gettempdir(), temp_name)

        images[0].save(temp_path, "JPEG")

        img = cv2.imread(temp_path)

        return img

    # ---------- IMAGE HANDLING ----------
    return cv2.imread(path)


# ---------------------------------------
# Feature Extraction
# ---------------------------------------
def extract_features_from_image(image_path):

    img = load_image(image_path)

    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image for consistent feature size
    gray = cv2.resize(gray, (512, 512))

    # ---------------------------------------
    # Image Quality Features
    # ---------------------------------------
    blur_score, noise_score = get_image_quality_metrics(gray)

    # ---------------------------------------
    # VISUAL FEATURES (HOG)
    # ---------------------------------------
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # Limit HOG size
    hog_features = hog_features[:300]

    # Normalize HOG
    hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-6)

    # ---------------------------------------
    # TEXT FEATURES (OCR)
    # ---------------------------------------
    text = pytesseract.image_to_string(gray)

    text_length = len(text.strip())

    # ---------------------------------------
    # LAYOUT FEATURES (Edges)
    # ---------------------------------------
    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.sum(edges > 0) / edges.size

    # ---------------------------------------
    # QR CODE DETECTION
    # ---------------------------------------
    qr_detector = cv2.QRCodeDetector()

    data, bbox, _ = qr_detector.detectAndDecode(img)

    qr_present = 1 if bbox is not None else 0

    # ---------------------------------------
    # COMBINE ALL FEATURES
    # ---------------------------------------
    features = np.hstack([
        hog_features,
        text_length / 1000,
        edge_density,
        blur_score / 1000,
        noise_score / 100,
        qr_present
    ])

    return features