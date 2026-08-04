from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_FILES = {
    "model": BASE_DIR / "Best_BrainTumor_Model.pkl",
    "scaler": BASE_DIR / "Scaler.pkl",
    "encoder": BASE_DIR / "LabelEncoder.pkl",
}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path: Path) -> np.ndarray:
    """Match the feature extraction used in ml.ipynb."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    gray = cv2.cvtColor(cv2.resize(image, (128, 128)), cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_features, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    lbp_features = lbp_features.astype(float)
    lbp_features /= lbp_features.sum() + 1e-7

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True)
    glcm_features = np.array(
        [
            graycoprops(glcm, "contrast")[0, 0],
            graycoprops(glcm, "dissimilarity")[0, 0],
            graycoprops(glcm, "homogeneity")[0, 0],
            graycoprops(glcm, "energy")[0, 0],
            graycoprops(glcm, "correlation")[0, 0],
            graycoprops(glcm, "ASM")[0, 0],
        ]
    )
    return np.concatenate([hog_features, lbp_features, glcm_features]).reshape(1, -1)


def load_artifacts():
    missing = [path.name for path in MODEL_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) + ". Export them from ml.ipynb first."
        )
    return tuple(joblib.load(path) for path in MODEL_FILES.values())


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = image_url = None
    if request.method == "POST":
        upload = request.files.get("image")
        if not upload or not upload.filename:
            error = "Please choose an MRI image to upload."
        elif not allowed_file(upload.filename):
            error = "Use a PNG, JPG, JPEG, BMP, TIF, or TIFF image."
        else:
            try:
                UPLOAD_DIR.mkdir(exist_ok=True)
                filename = secure_filename(upload.filename)
                image_path = UPLOAD_DIR / filename
                upload.save(image_path)

                model, scaler, encoder = load_artifacts()
                features = scaler.transform(extract_features(image_path))
                prediction = model.predict(features)
                label = encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(model.predict_proba(features)) * 100)
                result = {"label": label, "confidence": f"{confidence:.2f}"}
                image_url = url_for("uploaded_file", filename=filename)
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html", result=result, error=error, image_url=image_url
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve an uploaded MRI so it can be shown with the prediction."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="Image must be 8 MB or smaller."), 413


if __name__ == "__main__":
    app.run(debug=True)
