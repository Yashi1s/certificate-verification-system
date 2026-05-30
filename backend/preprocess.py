import os

UPLOAD_DIR = "uploads"

def preprocess_certificate(filename):
    """
    Placeholder preprocessing function.
    Currently just checks if file exists.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}

    return {
        "status": "success",
        "message": "Certificate ready for preprocessing",
        "file_path": file_path
    }
