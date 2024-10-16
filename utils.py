from PIL import Image

def getDepthMapBBoxArea(image_depth_map: list[list], bounding_box_coordinates: list) -> Image:
    '''Ritorna la regione dell'immagine descritta dalle coordinate della bounding box'''

    I = image_depth_map
    b = bounding_box_coordinates

    height_start    = int(b[1]*I.shape[0])
    height_end      = int((b[1]+b[3])*I.shape[0])
    width_start     = int(b[0]*I.shape[1])
    width_end       = int((b[0]+b[2])*I.shape[1])

    region_of_interest = I[height_start:height_end, width_start:width_end]

    return Image.fromarray(region_of_interest)