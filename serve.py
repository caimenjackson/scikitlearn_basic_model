from waitress import serve
from flask_api import app  # Assuming 'app' is the Flask app object in 'flask-api.py'

serve(app, host='0.0.0.0', port=5000)
