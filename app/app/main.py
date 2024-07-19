"""
Created on Sat Mar 27 12:45:25 2021

@author: kluerman
"""
from flask import Flask, render_template
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'xyz'
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'csv', 'log'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def input():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8080)
