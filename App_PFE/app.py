
"""le fichier app.py"""

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import logging
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'JPG'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

MODEL_DETECTION_PATH = './Detection.pt'
MODEL_VARIETY_PATH = './Variety.pt'
MODEL_MATURITY_PATH = './Maturity.pt'

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load a YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

# Load models with error handling
model_detection = load_model(MODEL_DETECTION_PATH)
model_variety = load_model(MODEL_VARIETY_PATH)
model_maturity = load_model(MODEL_MATURITY_PATH)

# Ensure the upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ProcessedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    detected_classes = db.Column(db.String(120), nullable=False)
    confidence_scores = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_type = db.Column(db.String(50), nullable=False)

def process_image(input_path, output_path, analysis_type):
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image {input_path}")

        logging.info(f"Processing image {input_path} for {analysis_type}")

        detected_classes = []
        confidence_scores = []

        if analysis_type == 'detection':
            model = model_detection
        elif analysis_type == 'variety':
            model = model_variety
        elif analysis_type == 'maturity':
            model = model_maturity
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        results = model(img)
        logging.info(f"Model results: {results}")

        # Font size for results
        font_size = 4  # Adjust this value to increase the font size
        font_thickness = 9  # Adjust this value for thicker text
        if analysis_type == 'detection':
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), font_thickness)
                    label = model_detection.names[cls_id]
                    combined_label = f"{label} / {conf:.2f}"
                    cv2.putText(img, combined_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
                    detected_classes.append(label)
                    confidence_scores.append(f"{conf:.2f}")
            else:
                logging.info(f"No detections made on image {input_path}")

        elif analysis_type in ['variety', 'maturity']:
            if results[0].probs is not None:
                probs = results[0].probs
                predicted_index = probs.top1
                label = model.names[predicted_index]
                conf = probs.top1conf
                combined_label = f"{label} / {conf:.2f}"
                cv2.putText(img, combined_label, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                detected_classes.append(label)
                confidence_scores.append(f"{conf:.2f}")
            else:
                logging.info(f"No predictions made on image {input_path}")

        cv2.imwrite(output_path, img)
        logging.info(f"Image processed and saved to {output_path}")
        logging.info(f"Detected classes: {detected_classes}")
        logging.info(f"Confidence scores: {confidence_scores}")

        return detected_classes, confidence_scores

    except Exception as e:
        logging.error(f"Failed to process image {input_path}: {str(e)}")
        return [], []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        file = request.files['file']
        analysis_type = request.form.get('analysis_type')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            detected_classes, confidence_scores = process_image(filepath, output_path, analysis_type)

            # Save to database
            new_image = ProcessedImage(
                filename=filename,
                detected_classes=','.join(detected_classes),
                confidence_scores=','.join(confidence_scores),
                analysis_type=analysis_type
            )
            db.session.add(new_image)
            db.session.commit()

            return redirect(url_for('show_result', filename=filename))
    return render_template('upload.html')

@app.route('/result/<filename>')
def show_result(filename):
    image_url = url_for('static', filename=f'outputs/{filename}')
    image = ProcessedImage.query.filter_by(filename=filename).first()
    return render_template('result.html', image_url=image_url, filename=image.filename, detected_classes=image.detected_classes, confidence_scores=image.confidence_scores, timestamp=image.timestamp)

@app.route('/history')
def history():
    images = ProcessedImage.query.order_by(ProcessedImage.timestamp.desc()).all()
    return render_template('history.html', images=images)

if __name__ == '__main__':
    if os.path.exists('database.db'):
        os.remove('database.db')
    with app.app_context():
        db.create_all()
    app.run(debug=True)