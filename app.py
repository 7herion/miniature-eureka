from flask import jsonify, Flask, request
import utils
from depth_anything import depth_calculation
import numpy as np

app = Flask(__name__)

@app.route("/distance-imagelist", methods=["POST"])
def distance_image_list():
    '''
    Endpoint per stimare la distanza di una lista di immagini

    Esempio di payload per la request:
    [
        {
            "image_path": "test/img__136.jpg",
            "bounding_box": [
                0.5697,
                0.3929,
                0.1525,
                0.2355
            ]
        }
    ]
    '''

    image_list = request.json
    result_list: list = []

    for item in image_list:
        item_bbox = item['bounding_box']
        item_path = item['image_path']

        depth_map = depth_calculation(item_path)

        b_box_depth_map = utils.getDepthMapBBoxArea(depth_map, item_bbox)
        distance_estimate = round(np.percentile(b_box_depth_map, 20), 2) # 20esimo percentile arrotondato a 2 decimali

        result_list.append({
            'bounding_box': item_bbox,
            'image_path': item_path,
            'distance': distance_estimate,
        })

    return jsonify(result_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0')