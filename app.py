```python
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import requests
from flask import Flask, render_template, request
from skimage.feature import (
    graycomatrix,
    graycoprops,
    hog,
    local_binary_pattern,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "model": BASE_DIR / "Best_BrainTumor_Model.pkl",
    "scaler": BASE_DIR / "Scaler.pkl",
    "encoder": BASE_DIR / "LabelEncoder.pkl",
}

BASE_URL = (
    "https://huggingface.co/poojasank/"
    "BrainTumorModel/resolve/main"
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff",
}

app = Flask(__name__)

# Maximum uploaded image size: 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ---------------------------------------------------------
# Download model files from Hugging Face
# ---------------------------------------------------------
def download_model_files():
    """Download model files from Hugging Face if they don't exist."""

    for name, path in MODEL_FILES.items():

        if path.exists():
            print(f"{path.name} already exists.")
            continue

        url = f"{BASE_URL}/{path.name}"

        print(f"Downloading: {url}")

        response = requests.get(
            url,
            timeout=300
        )

        response.raise_for_status()

        with open(path, "wb") as f:
            f.write(response.content)

        print(
            f"Downloaded {path.name} "
            f"({path.stat().st_size} bytes)"
        )


# ---------------------------------------------------------
# Check allowed image extensions
# ---------------------------------------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# Extract features directly from image
# ---------------------------------------------------------
def extract_features(image):

    if image is None:
        raise ValueError("Uploaded image is invalid.")

    # Resize and convert to grayscale
    gray = cv2.cvtColor(
        cv2.resize(image, (128, 128)),
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # HOG Features
    # -----------------------------------------------------
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # -----------------------------------------------------
    # LBP Features
    # -----------------------------------------------------
    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform",
    )

    lbp_features, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )

    lbp_features = lbp_features.astype(float)

    lbp_features /= (
        lbp_features.sum() + 1e-7
    )

    # -----------------------------------------------------
    # GLCM Features
    # -----------------------------------------------------
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_features = np.array(
        [
            graycoprops(
                glcm,
                "contrast"
            )[0, 0],

            graycoprops(
                glcm,
                "dissimilarity"
            )[0, 0],

            graycoprops(
                glcm,
                "homogeneity"
            )[0, 0],

            graycoprops(
                glcm,
                "energy"
            )[0, 0],

            graycoprops(
                glcm,
                "correlation"
            )[0, 0],

            graycoprops(
                glcm,
                "ASM"
            )[0, 0],
        ]
    )

    # -----------------------------------------------------
    # Combine all features
    # -----------------------------------------------------
    features = np.concatenate(
        [
            hog_features,
            lbp_features,
            glcm_features,
        ]
    )

    return features.reshape(1, -1)


# ---------------------------------------------------------
# Load ML model, scaler and encoder
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def load_artifacts():

    print("Checking model files...")

    download_model_files()

    print("Loading model...")

    model = joblib.load(
        MODEL_FILES["model"]
    )

    print("Loading scaler...")

    scaler = joblib.load(
        MODEL_FILES["scaler"]
    )

    print("Loading encoder...")

    encoder = joblib.load(
        MODEL_FILES["encoder"]
    )

    print(
        "All model files loaded successfully."
    )

    return model, scaler, encoder


# ---------------------------------------------------------
# Home / Prediction page
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        # IMPORTANT:
        # Your HTML form uses name="image"
        upload = request.files.get("image")

        if not upload or upload.filename == "":
            error = "Please choose an MRI image."

        elif not allowed_file(upload.filename):
            error = (
                "Only PNG, JPG, JPEG, BMP, TIFF "
                "images are allowed."
            )

        else:

            try:

                # -------------------------------------------------
                # DO NOT SAVE FILE TO /uploads
                #
                # Vercel has a read-only filesystem.
                # Read uploaded image directly into memory.
                # -------------------------------------------------

                file_bytes = np.frombuffer(
                    upload.read(),
                    np.uint8
                )

                image = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    raise ValueError(
                        "Uploaded image could not be decoded."
                    )

                print(
                    "Image decoded successfully."
                )

                # -------------------------------------------------
                # Load model
                # -------------------------------------------------
                model, scaler, encoder = (
                    load_artifacts()
                )

                # -------------------------------------------------
                # Extract image features
                # -------------------------------------------------
                features = extract_features(
                    image
                )

                # -------------------------------------------------
                # Scale features
                # -------------------------------------------------
                features = scaler.transform(
                    features
                )

                # -------------------------------------------------
                # Prediction
                # -------------------------------------------------
                prediction = model.predict(
                    features
                )

                label = encoder.inverse_transform(
                    prediction
                )[0]

                # -------------------------------------------------
                # Confidence
                # -------------------------------------------------
                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            features
                        )
                    )

                    confidence = float(
                        np.max(
                            probabilities
                        ) * 100
                    )

                else:
                    confidence = 0.0

                # -------------------------------------------------
                # Result
                # -------------------------------------------------
                result = {
                    "label": str(label),
                    "confidence": (
                        f"{confidence:.2f}"
                    ),
                }

                print(
                    f"Prediction: {label}"
                )

                print(
                    f"Confidence: "
                    f"{confidence:.2f}%"
                )

            except Exception as e:

                import traceback

                traceback.print_exc()

                error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=None,
    )


# ---------------------------------------------------------
# File size error
# ---------------------------------------------------------
@app.errorhandler(413)
def too_large(error):

    return render_template(
        "index.html",
        error=(
            "Image size must be less than 8 MB."
        ),
        result=None,
        image_url=None,
    ), 413


# ---------------------------------------------------------
# Application startup
# ---------------------------------------------------------
print(
    "========== APP STARTING =========="
)

try:

    load_artifacts()

    print(
        "========== MODEL READY =========="
    )

except Exception:

    import traceback

    traceback.print_exc()


# ---------------------------------------------------------
# Run Flask locally
# ---------------------------------------------------------
if __name__ == "__main__":

    app.run(
        debug=True
    )
```
