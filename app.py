import os

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from predictor import predict_image

# ==========================================================
# Flask App Configuration
# ==========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# Helper Function
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# Prediction API
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # No file uploaded
    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "No image uploaded."
        }), 400

    file = request.files["image"]

    # Empty filename
    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "Please choose an MRI image."
        }), 400

    # Invalid extension
    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "message": "Unsupported file format."
        }), 400

    try:

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        prediction, confidence, probabilities = predict_image(filepath)

        return jsonify({

            "success": True,

            "prediction": prediction,

            "confidence": float(confidence),

            "probabilities": probabilities,

            "image": filepath.replace("\\", "/")

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
