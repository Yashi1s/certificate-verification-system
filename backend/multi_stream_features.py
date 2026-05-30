import cv2
import numpy as np
import pytesseract
#from pyzbar.pyzbar import decode

def extract_features(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- BLUR ----------
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # ---------- NOISE ----------
    noise = cv2.fastNlMeansDenoising(gray)
    noise_score = np.mean(np.abs(gray-noise))

    # ---------- EDGE ----------
    edges = cv2.Canny(gray,100,200)
    edge_density = np.sum(edges)/(gray.shape[0]*gray.shape[1])

    # ---------- OCR TEXT ----------
    text = pytesseract.image_to_string(gray)
    text_len = len(text)

    # ---------- QR CODE ----------
    qr_present = 0

    # ---------- LAYOUT ----------
    height,width = gray.shape
    layout_ratio = width/height

    return np.array([
        blur_score,
        noise_score,
        edge_density,
        text_len,
        qr_present,
        layout_ratio
    ])