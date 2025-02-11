from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from PIL import Image
import io
import os
from ultralytics import YOLO

app = Flask(__name__)

model_path = r"C:\Users\Istyak\Documents\Aircraft\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

OUTPUT_DIR = r"C:\Users\Istyak\Documents\Aircraft\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        file_extension = os.path.splitext(file.filename)[-1].lower()
        is_video = file_extension in [".mp4", ".avi", ".mov", ".mkv"]

        if not is_video:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))

            results = model(image)

            result = results[0]
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            detections = []
            for i, (cls_name, conf) in enumerate(zip(class_names, confidences)):
                detections.append({
                    "class": cls_name,
                    "confidence": float(conf),
                })

            result_image = result.plot()
            result_image_pil = Image.fromarray(result_image)

            output_filename = f"processed_{file.filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            result_image_pil.save(output_path, format="JPEG")

            return jsonify({
                "detections": detections,
                "image_url": f"/output/{output_filename}"
            }), 200

        else:
            video_path = os.path.join(OUTPUT_DIR, file.filename)
            file.save(video_path)

            results = model.predict(source=video_path, save=True, save_txt=True)

            processed_video_path = os.path.join(OUTPUT_DIR, "predict", file.filename)

            return jsonify({
                "detections": "Processed video successfully",
                "video_url": f"/output/predict/{file.filename}"
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/output/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
