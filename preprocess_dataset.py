import os
import cv2

# Paths
BASE_DIR = "dataset"
INPUT_DIRS = {
    "original": os.path.join(BASE_DIR, "original"),
    "forged": os.path.join(BASE_DIR, "forged")
}

OUTPUT_DIRS = {
    "original": os.path.join(BASE_DIR, "processed", "original"),
    "forged": os.path.join(BASE_DIR, "processed", "forged")
}

# Create output folders if not exist
for folder in OUTPUT_DIRS.values():
    os.makedirs(folder, exist_ok=True)

IMAGE_SIZE = (224, 224)

def preprocess_and_save(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, IMAGE_SIZE)

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize
            img = img / 255.0

            # Save processed image
            cv2.imwrite(output_path, (img * 255).astype("uint8"))

            print(f"Processed: {output_path}")

# Run preprocessing
print("Preprocessing ORIGINAL certificates...")
preprocess_and_save(INPUT_DIRS["original"], OUTPUT_DIRS["original"])

print("Preprocessing FORGED certificates...")
preprocess_and_save(INPUT_DIRS["forged"], OUTPUT_DIRS["forged"])

print("✅ Preprocessing completed successfully")
