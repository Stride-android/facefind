import base64
import os
import threading
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, request, render_template, jsonify
from flask import session, redirect, url_for, escape
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import face_recognition
import cv2
from threading import Lock

app = Flask(__name__)
# run_with_ngrok(app)

# Set the allowed file extensions for image and video files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mkv'}

# Configure the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS


# create message object instance
# msg = MIMEMultipart()
#
# # setup the parameters of the message
# sender = 'courseappmail@gmail.com'
# receiver = 'pushkarshinalkar@gmail.com'
# msg['From'] = sender
# msg['To'] = receiver
# msg['Subject'] = 'Missing Person Notification'
#
# # add text message to the email body
# body = 'Please Respond if this is the person you are looking for'
# msg.attach(MIMEText(body, 'plain'))
#
# # open the image file and attach it to the message
# with open('6.jpg', 'rb') as f:
#     img_data = f.read()
# img = MIMEImage(img_data, name='6.jpg')
# msg.attach(img)
#
# # create SMTP session
# server = smtplib.SMTP('smtp.gmail.com', 587)
# server.starttls()
# server.login(sender, 'bxqxsvjgdmpizpzb')
#
# # send the message via the server
# server.sendmail(sender, receiver, msg.as_string())
#
# # terminate the SMTP session
# server.quit()


# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


stop_video = None
stop_video_lock = threading.Lock()
nameofgud = ''
emailofgud = ''
polltext = '0 Matches Found '
polltextlive = '0 Matches Found '

pollframe = ''
pollframelive = ''


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data['email']
    name = data['name']
    global nameofgud
    global emailofgud
    nameofgud = name
    emailofgud = email
    return ('', 200)



@app.route('/success')
def success():
    guardian_name = session.get('guardian_name')
    guardian_email = session.get('guardian_email')
    return f"Guardian name: {guardian_name}, Guardian email: {guardian_email}"




@app.route('/handle_button_click', methods=['POST'])
def handle_button_click():
    data = request.get_json()
    if data['button_clicked']:
        global stop_video
        stop_video = True
        return 'Button clicked!'


# Assume you have a list of users with email and password information stored as tuples
users = [("pushkarshinalkar@gmail.com", "shinalkar"), ("user2@example.com", "password2")]


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']
    # Check if email and password match a user in the local variable
    for user in users:
        if email == user[0] and password == user[1]:
            # If the email and password match, return a success message
            return ('', 200)
    else:
        # If the email and password do not match, return an error message
        return ('Invalid email or password', 400)


# Define a route for polling
@app.route('/poll', methods=['GET'])
def poll():
    # Return a JSON response with the data to be polled
    global polltext
    data = {'message': polltext}
    return jsonify(data)


# Define a route for polling
@app.route('/poll2', methods=['GET'])
def poll2():
    global pollframe
    if pollframe != '':
        data = {'frame': pollframe}
        return jsonify(data)
    else:
        # Handle case where pollframe is empty or not defined
        return jsonify({'error': 'No frame data available'})


# Define a route for polling
@app.route('/polllive', methods=['GET'])
def polllive():
    # Return a JSON response with the data to be polled
    global polltextlive
    data = {'message': polltextlive}
    return jsonify(data)


# Define a route for polling
@app.route('/poll2live', methods=['GET'])
def poll2live():
    global pollframelive
    if pollframelive != '':
        data = {'frame': pollframelive}
        return jsonify(data)
    else:
        # Handle case where pollframe is empty or not defined
        return jsonify({'error': 'No frame data available'})


# Route for the home page
@app.route('/')
def home():
    return render_template('login.html')


@app.route('/home')
def homem():
    return render_template('home.html')


@app.route('/detection')
def page2():
    return render_template('detection.html')


@app.route('/viddetect')
def page3():
    return render_template('video_detect.html')


@app.route('/livedetect')
def page5():
    return render_template('live_detect.html')


@app.route('/register')
def page4():
    return render_template('register.html')


# Route for handling file upload and recorded face recognition
@app.route('/upload', methods=['POST'])
def upload():
    global stop_video
    stop_video = None
    # Check if the files are present in the request
    if 'images[]' not in request.files or 'video' not in request.files:
        return 'Error: Images and video file not found in the request'

    # Get the uploaded files
    images = request.files.getlist('images[]')
    video = request.files['video']

    # Process the uploaded images
    known_faces = []
    for image in images:
        if image and allowed_file(image.filename):
            print('running1')
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)

    # Process the uploaded video
    if video and allowed_file(video.filename):
        print('running')
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        if stop_video is None:
            while cap.isOpened():
                ret, frame = cap.read()
                # print('running3')
                if stop_video is True:
                    break
                if not ret:
                    break
                # Detect faces in the video frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(
                    frame, face_locations)

                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Compare the detected face with known faces
                    matches = face_recognition.compare_faces(
                        known_faces, face_encoding)

                    # Calculate the percentage of face match
                    face_match_percentage = matches.count(
                        True) / len(matches) * 100

                    # Draw a rectangle around the detected face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 255, 0), 2)

                    # Display the percentage of face match under the detected face
                    cv2.putText(frame, f"Match: {face_match_percentage:.2f}%", (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display the processed frame
                    print(face_match_percentage)
                    if face_match_percentage > 98.0:
                        global polltext
                        polltext = 'Person found with match percentage ' + str(face_match_percentage)
                        print('Match found')
                        # Save the frame as an image
                        frame_filename = f"frame_{top}_{right}_{bottom}_{left}.jpg"
                        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                        cv2.imwrite(frame_path, frame)

                    _, buffer = cv2.imencode('.jpg', frame)
                    global pollframe
                    pollframe = base64.b64encode(buffer).decode('utf-8')
                    cv2.waitKey(1) == ord('q')  # Add a delay to allow time for display
            # return render_template('index.html', frame=frame_base64)
        cap.release()
        cv2.destroyAllWindows()
    return 'Face recognition complete'


# Route for handling file upload and recorded face recognition
@app.route('/uploadlive', methods=['POST'])
def uploadlive():
    global stop_video
    stop_video = None
    # Check if the files are present in the request
    if 'images[]' not in request.files:
        return 'Error: Images and video file not found in the request'

    # Get the uploaded files
    images = request.files.getlist('images[]')

    # Process the uploaded images
    known_faces = []
    for image in images:
        if image and allowed_file(image.filename):
            print('running1')
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture('http://178.239.225.102/mjpg/video.mjpg?compression=50')

        print(stop_video)
        if stop_video is None:
            while True:
                ret, frame = cap.read()
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                print('accesing')
                print(stop_video)
                if stop_video is True:
                    break
                if not ret:
                    break
                # Detect faces in the video frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(
                    frame, face_locations)

                for face_location, face_encoding in zip(face_locations, face_encodings):
                    print('reached end')
                    # Compare the detected face with known faces
                    matches = face_recognition.compare_faces(
                        known_faces, face_encoding)

                    # Calculate the percentage of face match
                    face_match_percentage = matches.count(
                        True) / len(matches) * 100

                    # Draw a rectangle around the detected face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 255, 0), 2)

                    # Display the percentage of face match under the detected face
                    cv2.putText(frame, f"Match: {face_match_percentage:.2f}%", (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display the processed frame
                    print(face_match_percentage)
                    if face_match_percentage > 98.0:
                        global polltextlive
                        global nameofgud
                        polltextlive = nameofgud + ' found with match percentage ' + str(face_match_percentage)
                        print('Match found')
                        # Save the frame as an image
                        frame_filename = f"frame_{top}_{right}_{bottom}_{left}.jpg"
                        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                        cv2.imwrite(frame_path, frame)

                _, buffer = cv2.imencode('.jpg', frame)
                # print('reached end')
                # cv2.imshow('heelo', frame)
                global pollframelive
                pollframelive = base64.b64encode(buffer).decode('utf-8')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Add a delay to allow time for display
            # return render_template('index.html', frame=frame_base64)
        cap.release()
        cv2.destroyAllWindows()
    return 'Face recognition complete'


if __name__ == '__main__':
    app.run()
