from flask import Flask
from flask_restful import Api
from flask_cors import CORS

import os

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
api = Api(app)

CORS(app)

import resources

api.add_resource(resources.Detect, '/detect')
api.add_resource(resources.SyncNames, '/sync_names')
api.add_resource(resources.SamPoints, '/sam_points')
api.add_resource(resources.SamBox, '/sam_box')
api.add_resource(resources.SamSetImage, '/sam_set_image')
