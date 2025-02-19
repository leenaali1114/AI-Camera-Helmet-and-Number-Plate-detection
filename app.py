from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import easyocr
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models and configurations
print("Loading models...")
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("YOLOv3 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv3 model: {str(e)}")
    raise

try:
    model = load_model('helmet-nonhelmet_cnn.h5')
    print("Helmet detection model loaded successfully")
except Exception as e:
    print(f"Error loading helmet model: {str(e)}")
    raise

try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized successfully")
except Exception as e:
    print(f"Error initializing EasyOCR: {str(e)}")
    raise

COLORS = [(0,255,0),(0,0,255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def helmet_or_nohelmet(helmet_roi):
    try:
        if helmet_roi is None or helmet_roi.size == 0:
            print("Empty helmet ROI")
            return None
            
        # Preprocess the image
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        # Apply some preprocessing to improve detection
        helmet_roi = cv2.GaussianBlur(helmet_roi, (3, 3), 0)
        helmet_roi = cv2.normalize(helmet_roi, None, 0, 255, cv2.NORM_MINMAX)
        
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi/255.0
        
        prediction = model.predict(helmet_roi)
        confidence = prediction[0][0]
        print(f"Helmet prediction confidence: {confidence}")
        
        # Only return a prediction if we're confident enough
        if abs(confidence - 0.5) > 0.2:  # Requires 70% confidence
            return int(confidence > 0.5)
        else:
            print("Low confidence prediction, skipping")
            return None
            
    except Exception as e:
        print(f"Error in helmet detection: {str(e)}")
        return None

def process_frame(frame):
    if frame is None:
        print("Empty frame received")
        return None, {'plates': [], 'helmet_status': []}

    print("Processing frame...")
    img = imutils.resize(frame, height=500)
    height, width = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []
    detected_info = {
        'plates': [],
        'helmet_status': []
    }
    
    # Store detections for combined display
    current_detections = {
        'plate': None,
        'helmet_status': None,
        'bike_position': None
    }

    # Detection loop
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.3:  # Increased confidence threshold
                print(f"Detection found: class {class_id} with confidence {confidence}")
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Number of detections after NMS: {len(indexes)}")

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            
            if classIds[i] == 0:  # bike
                print("Processing bike detection")
                current_detections['bike_position'] = (x, y, w, h)
                
                # Adjust the helmet ROI to focus on the upper part of the bike
                helmet_y = max(0, y - int(h * 0.2))
                helmet_h = int(h * 0.4)
                helmet_roi = img[helmet_y:helmet_y+helmet_h, x:x+w]
                
                # Draw the helmet detection region (blue rectangle)
                cv2.rectangle(img, (x, helmet_y), (x + w, helmet_y + helmet_h), (255, 0, 0), 2)
                
                helmet_status = helmet_or_nohelmet(helmet_roi)
                
                if helmet_status is not None:
                    status = "Helmet" if helmet_status == 0 else "No Helmet"
                    color = (0, 255, 0) if helmet_status == 0 else (0, 0, 255)
                    print(f"Helmet status: {status}")
                    detected_info['helmet_status'].append(status)
                    current_detections['helmet_status'] = (status, color)
                
                # Draw bike detection (green rectangle)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elif classIds[i] == 1:  # number plate
                print("Processing number plate detection")
                number_plate_roi = img[y:y+h, x:x+w]
                if number_plate_roi.size > 0:
                    number_plate_roi = cv2.resize(number_plate_roi, 
                                                (number_plate_roi.shape[1]*2, 
                                                 number_plate_roi.shape[0]*2))
                    
                    results = reader.readtext(number_plate_roi)
                    if results:
                        plate_text = results[0][1]
                        print(f"Detected plate: {plate_text}")
                        detected_info['plates'].append(plate_text)
                        current_detections['plate'] = plate_text
                
                # Draw number plate detection (red rectangle)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display combined information at the bottom
        if current_detections['plate'] or current_detections['helmet_status']:
            # Create combined text
            info_text = []
            if current_detections['plate']:
                info_text.append(f"Plate: {current_detections['plate']}")
            if current_detections['helmet_status']:
                status, color = current_detections['helmet_status']
                info_text.append(f"Status: {status}")
            
            combined_text = " | ".join(info_text)
            
            # Calculate position for bottom display
            text_size = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2  # Center horizontally
            text_y = height - 20  # 20 pixels from bottom
            
            # Draw background rectangle
            padding = 10
            cv2.rectangle(img,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(img, combined_text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    print(f"Final detection info: {detected_info}")
    return img, detected_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process based on file type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return process_image(filepath)
        else:
            return process_video(filepath)
    
    return jsonify({'error': 'Invalid file type'})

def process_image(filepath):
    frame = cv2.imread(filepath)
    processed_frame, detection_info = process_frame(frame)
    
    # Save processed image
    output_filename = 'processed_' + os.path.basename(filepath)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, processed_frame)
    
    return jsonify({
        'result_image': url_for('static', filename=f'uploads/{output_filename}'),
        'detections': detection_info
    })

def process_video(filepath):
    output_filename = 'processed_' + os.path.basename(filepath)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    all_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, detection_info = process_frame(frame)
        out.write(processed_frame)
        all_detections.append(detection_info)
    
    cap.release()
    out.release()
    
    return jsonify({
        'result_video': url_for('static', filename=f'uploads/{output_filename}'),
        'detections': all_detections
    })

@app.route('/default_video')
def default_video():
    def generate_frames():
        cap = cv2.VideoCapture('video.mp4')  # Your default video path
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                processed_frame, detection_info = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
