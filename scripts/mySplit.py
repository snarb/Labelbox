import json
import random
import numpy as np
from random import shuffle
import copy
from sympy import Point, Polygon

testRate = 0.11

dir = r'/home/brans/repos/Labelbox/scripts/'



with open(dir + r'train.json', encoding='utf-8') as data_file:

    jsonObject = json.loads(data_file.read())

    jsonTest = copy.deepcopy(jsonObject)
    jsonTrain = copy.deepcopy(jsonObject)

    imgs = jsonObject['images']

    sizes = {}

    for img in imgs:
        sizes[img['id']] = (img['width'], img['height'])

    shuffle(imgs)
    testSize = int(testRate * len(imgs))

    imgsTest = imgs[:testSize]
    imgsTrain = imgs[testSize:]

    imgsTestIds = set()
    for img in imgsTest:
        imgsTestIds.add(img['id'])

    jsonTest['images'] = imgsTest
    jsonTrain['images'] = imgsTrain

    anots = jsonObject['annotations']
    anotsTest = []
    anotsTrain = []

    for anot in anots:
        mask = anot['segmentation'][0]
        mask = [int(max(x, 0)) for x in mask]
        anot['segmentation'][0] = mask
        it = iter(mask)
        cc = list(zip(it, it))
        pl = Polygon(*cc)
        imageId = anot['image_id']
        widthMax, heightMax = sizes[imageId]
        x = int(pl.bounds[0])
        y = int(pl.bounds[1])

        width = int(pl.bounds[2] - x)
        assert(width > 0)
        difW = widthMax - x - width

        assert(difW > -100)
        if(difW < 0):
            width += difW

        height = int(pl.bounds[3] - y)
        assert (height > 0)
        difH = heightMax - y - height
        assert (difH > -100)
        if(difH < 0):
            height += difH

        anot['bbox'] =  [x, y, width, height]


        if(imageId in imgsTestIds):
            anotsTest.append(anot)
        else:
            anotsTrain.append(anot)

    jsonTest['annotations'] = anotsTest
    jsonTrain['annotations'] = anotsTrain

    #json_raws = data_file.readlines()

    with open(dir + r'train_2.json', 'w') as fp:
        json.dump(jsonTrain, fp)

    with open(dir + r'validation.json', 'w') as fp:
        json.dump(jsonTest, fp)
