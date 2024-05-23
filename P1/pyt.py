import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
import os

# Initialize Flask app
app = Flask(__name__, template_folder="C:\\Users\\SMASH\\Downloads\\P1\\templates")

# Google Sheets setup
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\SMASH\Downloads\P1\credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open_by_key('1ynNwpqFrm5fhX7lMN0uihtS2VmmKaQzw4FJKn6It4zw').sheet1

# Google Drive folder for image uploads
drive_folder_url = 'https://drive.google.com/drive/folders/1wAO1_UjxiIemXZkRW1MuU7HiqXwzZ0WJ?usp=drive_link'

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
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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

# Upload image to Google Drive and return its URL
def upload_image_to_drive(image_data):
    # Implement logic to upload image to Google Drive
    # Return the URL of the uploaded image
    # For now, return a placeholder URL
    return drive_folder_url + '/placeholder.jpg'

# Add registration data and image URL to Google Sheet
def add_data_to_sheet(reg_data, image_url):
    sheet.append_row(reg_data + [image_url])

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
            if 'imageUrl' not in request.files:
                return "No file part"

            image_data = request.files['imageUrl']
            if image_data.filename == '':
                return "No selected file"

            # Securely save the file to the upload folder
            image_url = upload_image_to_drive(image_data)

            # Retrieve other registration data from the form
            reg_data = [
                request.form['regNumber'],
                request.form['regOwner'],
                request.form['address'],
                request.form['makersClass'],
                request.form['vehicleClass'],
                request.form['manufactureDate'],
                request.form['dateOfRegistration'],
                request.form['regVaild'],
                request.form['capacity'],
                request.form['fuelType'],
                request.form['state']
            ]

            # Add registration data and image URL to Google Sheet
            add_data_to_sheet(reg_data, image_url)

            return redirect(url_for('registration'))
        except Exception as e:
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
        date_of_registration = request.form['dateOfRegistration']
        reg_valid = request.form['regValid']
        capacity = request.form['capacity']
        fuel_type = request.form['fuelType']
        state = request.form['state']
        image_url = request.form['imageUrl']  # Assuming image URL is also submitted via form
        return "Form submitted successfully"
    return render_template('registration.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        try:
            # Retrieve form data
            reg_data = [
                request.form['regNumber'],
                request.form['regOwner'],
                request.form['address'],
                request.form['makersClass'],
                request.form['vehicleClass'],
                request.form['manufactureDate'],
                request.form['dateOfRegistration'],
                request.form['regVaild'],
                request.form['capacity'],
                request.form['fuelType'],
                request.form['state']
            ]
            image_url = request.files['imageUrl'].filename  # Assuming image URL is also submitted via form

            # Add registration data and image URL to Google Sheet
            add_data_to_sheet(reg_data, image_url)

            return jsonify({'message': 'Success'})  # Return success message as JSON
        except Exception as e:
            return jsonify({'error': str(e)})  # Return error message as JSON
    else:
        return jsonify({'error': 'Invalid request'})  # Return error message for invalid request

if __name__ == '__main__':
    app.run(debug=True)
