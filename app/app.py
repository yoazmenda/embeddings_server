from flask import Flask, render_template, request, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_health', methods=['GET'])
def generate():
    return "healty and running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
