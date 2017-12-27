# #!flask/bin/python
# import random
# import string
#
# from datetime import timedelta
# from functools import update_wrapper
#
# from flask import Flask, request, jsonify, current_app, make_response, render_template
#
# from utils.make_predictions import *
#
#
# def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
#                 attach_to_all=True, automatic_options=True):
#     """Decorator function that allows crossdomain requests.
#       Courtesy of
#       https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
#     """
#     if methods is not None:
#         methods = ', '.join(sorted(x.upper() for x in methods))
#     if headers is not None and not isinstance(headers, basestring):
#         headers = ', '.join(x.upper() for x in headers)
#     if not isinstance(origin, basestring):
#         origin = ', '.join(origin)
#     if isinstance(max_age, timedelta):
#         max_age = max_age.total_seconds()
#
#     def get_methods():
#         """ Determines which methods are allowed
#         """
#         if methods is not None:
#             return methods
#
#         options_resp = current_app.make_default_options_response()
#         return options_resp.headers['allow']
#
#     def decorator(f):
#         """The decorator function
#         """
#         def wrapped_function(*args, **kwargs):
#             """Caries out the actual cross domain code
#             """
#             if automatic_options and request.method == 'OPTIONS':
#                 resp = current_app.make_default_options_response()
#             else:
#                 resp = make_response(f(*args, **kwargs))
#             if not attach_to_all and request.method != 'OPTIONS':
#                 return resp
#
#             h = resp.headers
#             h['Access-Control-Allow-Origin'] = origin
#             h['Access-Control-Allow-Methods'] = get_methods()
#             h['Access-Control-Max-Age'] = str(max_age)
#             h['Access-Control-Allow-Credentials'] = 'true'
#             h['Access-Control-Allow-Headers'] = \
#                 "Origin, X-Requested-With, Content-Type, Accept, Authorization"
#             if headers is not None:
#                 h['Access-Control-Allow-Headers'] = headers
#             return resp
#
#         f.provide_automatic_options = False
#         return update_wrapper(wrapped_function, f)
#     return decorator
#
# set_workspace("data/pagoda")
# mean_proto = workspace("data/mean.binaryproto")
# caffe_deploy = workspace("caffe_model/caffenet_deploy.prototxt")
#
# py_render_template("template/caffenet_deploy.template", caffe_deploy)
#
# mean_data = read_mean_data(mean_proto)
# net = read_model_and_weight(caffe_deploy, workspace("caffe_model/snapshot_iter_15000.caffemodel"))
# transformer = image_transformers(net, mean_data)
#
# app = Flask(__name__, template_folder="web")
#
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# if not os.path.isdir(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
#
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
#
#
# @app.route('/')
# def index():
#     return render_template('visualize.html')
#
#
# @app.route('/predict', methods = ['POST'])
# @crossdomain(origin='*')
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
#             file_name = file_name + "." + file.filename.rsplit('.', 1)[1]
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#             file.save(file_path)
#             predict = single_making_prediction(file_path, transformer, net)
#             os.remove(file_path)
#             return jsonify(predict.tolist())
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)
