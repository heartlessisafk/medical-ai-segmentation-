import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from predict import run_inference

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)

MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Please choose an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Unsupported file type. Allowed: png, jpg, jpeg, bmp, tif, tiff.",
            }
        ), 400

    if not MODEL_PATH.exists():
        return jsonify(
            {
                "success": False,
                "error": (
                    "Model checkpoint not found at checkpoints/best_model.pth. "
                    "Train the model first or add a valid checkpoint."
                ),
            }
        ), 500

    try:
        original_name = secure_filename(file.filename)
        sample_id = uuid.uuid4().hex[:12]
        extension = original_name.rsplit(".", 1)[1].lower()

        upload_path = UPLOAD_DIR / f"{sample_id}_input.{extension}"
        file.save(upload_path)

        with Image.open(upload_path) as img:
            img.verify()

        result = run_inference(
            image_path=str(upload_path),
            checkpoint_path=str(MODEL_PATH),
            output_dir=str(OUTPUT_DIR),
            sample_id=sample_id,
        )

        response = {
            "success": True,
            "original_image": f"/outputs/{Path(result['original_image']).name}",
            "mask_image": f"/outputs/{Path(result['mask_image']).name}",
            "overlay_image": f"/outputs/{Path(result['overlay_image']).name}",
            "confidence": round(float(result["confidence"]), 4),
            "coverage_percent": round(float(result["coverage_percent"]), 2),
            "threshold": float(result["threshold"]),
        }
        return jsonify(response), 200

    except Exception as exc:
        return jsonify({"success": False, "error": f"Inference failed: {str(exc)}"}), 500


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)