from flask import Flask, render_template, Response
import cv2
from helper import realtime
import pandas as pd

app = Flask(__name__)


# def generate_frames():
#     image, result = realtime()
#     ret, buffer = cv2.imencode('.jpg', image)
#     image = buffer.tobytes()

#     yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(realtime(1), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/testdata', methods=['POST', 'GET'])
def testdata():
    return render_template('testdata.html')


if __name__ == "__main__":
    app.run(debug=True)
