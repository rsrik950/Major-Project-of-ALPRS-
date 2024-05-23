from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__, template_folder="C:\\Users\\SMASH\\Downloads\\P1\\templates")
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\SMASH\\Downloads\\P1\\static\\assets\\uploads'

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load pre-trained models
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Load YOLO
net = cv2.dnn.readNet(r'C:\Users\SMASH\Downloads\yolov3.weights', r'C:\Users\SMASH\Downloads\yolov3.cfg')

# Check if YOLO is loaded properly
if net.empty():
    print("Error: Failed to load YOLO network.")
    exit()

# Define the indices of the unconnected output layers
unconnected_out_layers_indices = [200, 227, 254]

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers_indices]

# ALPR function
def alpr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in plates:
        plate_img = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(plate_img)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

# YOLO function
def yolo_detection(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

# Process image function
def process_image(frame):
    frame = alpr(frame)
    frame = yolo_detection(frame)
    return frame

# Video capturing function
def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_image(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second_page():
    return render_template('second.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            # Check if 'image' key exists in the request.files dictionary
            if 'image' not in request.files:
                return "No file part"
            
            # Retrieve the file from the request
            image_file = request.files['image']
            
            # If the user does not select a file, the browser submits an empty part
            if image_file.filename == '':
                return "No selected file"

            # Securely save the file to the upload folder
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Redirect to another route passing the filename as an argument
            return redirect(url_for('registration', filename=filename))
        except Exception as e:
            # Handle other exceptions if necessary
            return f"An error occurred: {str(e)}"
    return render_template('upload.html')

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        reg_number = request.form['regNumber']
        reg_owner = request.form['regOwner']
        address = request.form['address']
        makers_class = request.form['makersClass']
        vehicle_class = request.form['vehicleClass']
        manufacture_date = request.form['manufactureDate']
        fuel_type = request.form['fuelType']
        state = request.form['state']
        return "Form submitted successfully"
    filename = request.args.get('filename')
    return render_template('registration.html', filename=filename)

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/process_image_data', methods=['POST'])
def process_image_data():
    # Retrieve the image data from the POST request
    image_data = request.files['image']
    
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    # Perform image processing tasks here
    if image_data:
        try:
            # Save the image to a folder
            filename = secure_filename(image_data.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_data.save(image_path)
            
            # Return a success response with the filename
            return jsonify({'message': 'Image saved successfully', 'filename': filename})
        except Exception as e:
            # Return an error response if saving fails
            return jsonify({'error': str(e)})
    
    # Return an error response if no image data is received
    return jsonify({'error': 'No image data received'})

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
