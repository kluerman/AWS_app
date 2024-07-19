"""
Created on Sat Mar 27 12:45:25 2021

@author: kluerman
"""
from flask import Flask, render_template
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/')
def input():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8080)
