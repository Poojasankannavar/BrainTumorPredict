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


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# FLASK APPLICATION
# IMPORTANT: Vercel looks for this top-level "app" object
# =========================================================

app = Flask(__name__)

# Maximum upload size = 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILES = {
    "model": BASE_DIR / "Best_BrainTumor_Model.pkl",
    "scaler": BASE_DIR / "Scaler.pkl",
    "encoder": BASE_DIR / "LabelEncoder.pkl",
}


# =========================================================
# HUGGING FACE MODEL LOCATION
# =========================================================

BASE_URL = (
    "https://huggingface.co/poojasank/"
    "BrainTumorModel/resolve/main"
)


# =========================================================
# ALLOWED IMAGE EXTENSIONS
# =========================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff",
}


# =========================================================
# DOWNLOAD MODEL FILES
# =========================================================

def download_model_files():
    """
    Download model files from Hugging Face
    if they are not already available locally.
    """

    for name, path in MODEL_FILES.items():

        if path.exists():
            print(
                f"{path.name} already exists."
            )
            continue

        url = f"{BASE_URL}/{path.name}"

        print(
            f"Downloading model file: {url}"
        )

        response = requests.get(
            url,
            timeout=300
        )

        response.raise_for_status()

        with open(path, "wb") as file:
            file.write(response.content)

        print(
            f"Downloaded {path.name}"
        )

        print(
            f"File size: "
            f"{path.stat().st_size} bytes"
        )


# =========================================================
# CHECK FILE EXTENSION
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    if image is None:
        raise ValueError(
            "Uploaded image is invalid."
        )

    # -----------------------------------------------------
    # Resize image
    # -----------------------------------------------------

    resized_image = cv2.resize(
        image,
        (128, 128)
    )

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # HOG FEATURES
    # =====================================================

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # =====================================================
    # LBP FEATURES
    # =====================================================

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
        bins=np.arange(
            0,
            n_points + 3
        ),
        range=(
            0,
            n_points + 2
        ),
    )

    lbp_features = (
        lbp_features.astype(float)
    )

    lbp_features /= (
        lbp_features.sum()
        + 1e-7
    )

    # =====================================================
    # GLCM FEATURES
    # =====================================================

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

    # =====================================================
    # COMBINE FEATURES
    # =====================================================

    features = np.concatenate(
        [
            hog_features,
            lbp_features,
            glcm_features,
        ]
    )

    return features.reshape(
        1,
        -1
    )


# =========================================================
# LOAD MODEL / SCALER / ENCODER
# =========================================================

@lru_cache(maxsize=1)
def load_artifacts():

    print(
        "Checking model files..."
    )

    # Download model files if required
    download_model_files()

    # -----------------------------------------------------
    # Load ML model
    # -----------------------------------------------------

    print(
        "Loading model..."
    )

    model = joblib.load(
        MODEL_FILES["model"]
    )

    # -----------------------------------------------------
    # Load scaler
    # -----------------------------------------------------

    print(
        "Loading scaler..."
    )

    scaler = joblib.load(
        MODEL_FILES["scaler"]
    )

    # -----------------------------------------------------
    # Load label encoder
    # -----------------------------------------------------

    print(
        "Loading encoder..."
    )

    encoder = joblib.load(
        MODEL_FILES["encoder"]
    )

    print(
        "All model files loaded successfully."
    )

    return (
        model,
        scaler,
        encoder
    )


# =========================================================
# HOME PAGE + PREDICTION
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None

    error = None

    if request.method == "POST":

        # -------------------------------------------------
        # Get uploaded image
        #
        # Your HTML input must use:
        #
        # <input type="file" name="image">
        # -------------------------------------------------

        upload = request.files.get(
            "image"
        )

        # -------------------------------------------------
        # Check whether image was selected
        # -------------------------------------------------

        if (
            not upload
            or upload.filename == ""
        ):

            error = (
                "Please choose an MRI image."
            )

        # -------------------------------------------------
        # Check image extension
        # -------------------------------------------------

        elif not allowed_file(
            upload.filename
        ):

            error = (
                "Only PNG, JPG, JPEG, BMP, "
                "and TIFF images are allowed."
            )

        else:

            try:

                # =================================================
                # IMPORTANT FOR VERCEL
                #
                # DO NOT USE:
                #
                # upload.save(...)
                #
                # DO NOT CREATE:
                #
                # /uploads
                #
                # Vercel filesystem is read-only.
                #
                # Read image directly into memory instead.
                # =================================================

                file_bytes = np.frombuffer(
                    upload.read(),
                    np.uint8
                )

                image = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                # -------------------------------------------------
                # Validate image
                # -------------------------------------------------

                if image is None:

                    raise ValueError(
                        "Uploaded image could not "
                        "be decoded."
                    )

                print(
                    "Image decoded successfully."
                )

                # =================================================
                # LOAD MODEL
                # =================================================

                (
                    model,
                    scaler,
                    encoder
                ) = load_artifacts()

                # =================================================
                # EXTRACT FEATURES
                # =================================================

                features = extract_features(
                    image
                )

                print(
                    "Features extracted successfully."
                )

                # =================================================
                # SCALE FEATURES
                # =================================================

                features = scaler.transform(
                    features
                )

                # =================================================
                # MAKE PREDICTION
                # =================================================

                prediction = model.predict(
                    features
                )

                # =================================================
                # CONVERT PREDICTION TO LABEL
                # =================================================

                label = (
                    encoder
                    .inverse_transform(
                        prediction
                    )[0]
                )

                # =================================================
                # CALCULATE CONFIDENCE
                # =================================================

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

                # =================================================
                # STORE RESULT
                # =================================================

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

    # =====================================================
    # RETURN HTML
    # =====================================================

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=None,
    )


# =========================================================
# FILE SIZE ERROR
# =========================================================

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


# =========================================================
# APPLICATION STARTUP
# =========================================================

print(
    "========== APP STARTING =========="
)


# IMPORTANT:
# Do NOT load the model during Vercel build/import.
#
# The model is loaded when prediction is requested.
#
# This helps avoid unnecessary startup problems.

print(
    "========== APP READY =========="
)


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
```
