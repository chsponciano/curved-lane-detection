from flask import Flask 
from flask_socketio import SocketIO
from action import initialize, get_angle, is_process
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
CORS(app)

@app.route('/init')
def init():
    initialize()
    return 'init'

@app.route('/angle')
def angle():
    if (is_process()):
        return str(int(get_angle()))
    else:
        return 'finish'

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0', threaded=True)