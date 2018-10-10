import json
import datetime as dt
import logging
from shapely import wkt
import requests
from PIL import Image
import random
from copy import deepcopy

def from_json(labeled_data, coco_output):
    # read labelbox JSON output
    with open(labeled_data, 'r') as f:
        lines = f.readlines()
        label_data = json.loads(lines[0])

    # setup COCO dataset container and info
        coco = {
            'info': None,
            'images': [],
            'annotations': [],
            'licenses': [],
            'categories': []
        }

        coco['info'] = {
            'year': dt.datetime.now(dt.timezone.utc).year,
            'version': None,
            'description': label_data[0]['Project Name'],
            'contributor': label_data[0]['Created By'],
            'url': 'labelbox.com',
            'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
        }

        cocoTest = deepcopy(coco)

        for data in label_data:
            # # Download and get image name
            # try:
            #     response = requests.get(data['Labeled Data'], stream=True)
            # except requests.exceptions.MissingSchema as e:
            #     logging.exception(('"Labeled Data" field must be a URL. '
            #                       'Support for local files coming soon'))
            #     continue
            # except requests.exceptions.ConnectionError as e:
            #     logging.exception('Failed to fetch image from {}'
            #                       .format(data['Labeled Data']))
            #     continue
            #
            # if(data['Label'] == 'Skip'):
            #     continue
            #
            # response.raw.decode_content = True

            if(data['Label'] == 'Skip'):
                continue

            #path = 'D:\\crades\\photos\\' + data['External ID']
            path = 'D:\\crades\\photosAll\\'  + data['External ID']
            im = Image.open(path)
            width, height = im.size

            image = {
                "id": data['ID'],
                "width": width,
                "height": height,
                "file_name": path,
                "license": None,
                "flickr_url": data['Labeled Data'],
                "coco_url": data['Labeled Data'],
                "date_captured": None,
            }

            coco['images'].append(image)

            # convert WKT multipolygon to COCO Polygon format
            for cat in data['Label'].keys():

                if('image_problems' not in data['Label']):

                    try:
                        # check if label category exists in 'categories' field
                        cat_id = [c['id'] for c in coco['categories']
                                  if c['supercategory'] == cat][0]
                    except IndexError as e:
                        cat_id = len(coco['categories']) + 1
                        category = {
                            'supercategory': cat,
                            'id': len(coco['categories']) + 1,
                            'name': cat
                        }
                        coco['categories'].append(category)

                    multipolygon = wkt.loads(data['Label'][cat])
                    for m in multipolygon:
                        segmentation = []
                        for x, y in m.exterior.coords:
                            segmentation.extend([x, height-y])

                        annotation = {
                            "id": len(coco['annotations']) + 1,
                            "image_id": data['ID'],
                            "category_id": cat_id,
                            "segmentation": [segmentation],
                            "area": m.area,  # float
                            "bbox": [m.bounds[0], m.bounds[1],
                                     m.bounds[2]-m.bounds[0],
                                     m.bounds[3]-m.bounds[1]],
                            "iscrowd": 0
                        }


                        curRnd = random.random()
                        if(curRnd < 0.0):
                            cocoTest['annotations'].append(annotation)
                        else:
                            coco['annotations'].append(annotation)


        with open(coco_output, 'w+') as f:
            f.write(json.dumps(coco))

        with open('validation.json', 'w+') as f:
            f.write(json.dumps(cocoTest))
