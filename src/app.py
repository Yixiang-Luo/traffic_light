import cv2
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template, url_for
import os

app = Flask(__name__)

model = YOLO("../runs/detect/train3/weights/last.pt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '../uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, '../static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result_file = None
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            results = model.predict(source=file_path, save=False)
            for result in results:
                img_array = result.plot()
                from PIL import Image
                img = Image.fromarray(img_array)
                result_file = os.path.join(RESULT_FOLDER, file.filename)
                img.save(result_file)

        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            result_file = os.path.join(RESULT_FOLDER, file.filename)
            cap = cv2.VideoCapture(file_path)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(
                result_file,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, save=False)
                result_frame = results[0].plot()

                out.write(result_frame)

            cap.release()
            out.release()

        else:
            return jsonify({"error": "Unsupported file format!"}), 400

        if result_file and os.path.exists(result_file):
            result_url = url_for('static', filename=f'results/{file.filename}')
            return jsonify({"result_url": result_url})

    return jsonify({"error": "No file uploaded!"}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
