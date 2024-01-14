from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'roi_du_monde'

#from app.routes import apps
#from app.routes import star
from app.routes import dashbord



