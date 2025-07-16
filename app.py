from flask import Flask, render_template, request
from core import process_url

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        url = request.form.get('url')
        message = process_url(url)
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
