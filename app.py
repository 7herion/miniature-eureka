from flask import jsonify, Flask, request
import utils
from depth_anything import depth_calculation
import numpy as np
import ast

app = Flask(__name__)

@app.route("/distance-imagelist", methods=["POST"])
def distance_image_list():
    '''Endpoint per stimare la distanza di una lista di immagini'''

    image_list = request.json

    for item in image_list:
        item_path = item['filename']

        depth_map = depth_calculation(item_path)

        for bounding_box in item['bb_box']:
            bbox_coordinates = ast.literal_eval(bounding_box['box'])

            b_box_depth_map = utils.getDepthMapBBoxArea(depth_map, bbox_coordinates)
            distance_estimate = round(np.percentile(b_box_depth_map, 20), 2) # 20esimo percentile arrotondato a 2 decimali

            bounding_box['distance'] = distance_estimate

    return jsonify(image_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0')