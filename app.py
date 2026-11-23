
import os
import re
import logging
import cv2
import numpy as np
import easyocr

from flask import Flask, request, jsonify
from supabase import create_client

app = Flask(__name__)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")
OCR_API_TOKEN = os.environ.get("OCR_API_TOKEN")

# Validate environment variables
if not SUPABASE_URL or not SUPABASE_KEY or not OCR_API_TOKEN:
    raise RuntimeError("Missing required environment variables")

# Supabase connection
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load OCR model once
reader = easyocr.Reader(["en"], gpu=False)

logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Electricity Meter Cloud OCR Server Running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/process", methods=["POST"])
def process_image():

    # API authentication
    if request.headers.get("X-API-KEY") != OCR_API_TOKEN:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}

    meter_id = data.get("meter_id")
    image_path = data.get("image_path")

    if not meter_id or not image_path:
        return jsonify({
            "success": False,
            "error": "Missing meter_id or image_path"
        }), 400

    try:
        # Download image from Supabase Storage
        image_bytes = (
            supabase.storage
            .from_("meter-images")
            .download(image_path)
        )

        # Convert image bytes
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        # OCR text extraction
        results = reader.readtext(
            image,
            allowlist="0123456789."
        )

        if not results:
            raise ValueError("No meter reading detected")

        # Combine detected digits
        reading_text = "".join(
            item[1] for item in results
        )

        reading_text = reading_text.replace(",", ".")

        match = re.search(r"\d+(?:\.\d+)?", reading_text)

        if not match:
            raise ValueError("No valid numeric reading")

        meter_value = float(match.group())

        confidence = max(float(item[2]) for item in results)

        # Prepare database record
        record = {
            "meter_id": str(meter_id),
            "meter_value": meter_value,
            "image_path": image_path,
            "status": "completed",
            "ocr_confidence": confidence
        }

        # Insert into Supabase database
        supabase.table("meter_readings").insert(record).execute()

        return jsonify({
            "success": True,
            "meter_id": meter_id,
            "meter_value": meter_value,
            "confidence": confidence
        }), 200

    except Exception:
        logging.exception("Meter OCR processing failed")

        return jsonify({
            "success": False,
            "error": "Image processing or database operation failed"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
