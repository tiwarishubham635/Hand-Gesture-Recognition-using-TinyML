from flask import Flask, render_template, Response
from helper import realtime

app = Flask(__name__)

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
