import json
import random
import numpy as np
from random import shuffle
import copy

testRate = 0.11


with open(r'D:\repos\Labelbox\scripts\train.json', encoding='utf-8') as data_file:

    jsonObject = json.loads(data_file.read())

    jsonTest = copy.deepcopy(jsonObject)
    jsonTrain = copy.deepcopy(jsonObject)

    imgs = jsonObject['images']


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
        imageId = anot['image_id']
        if(imageId in imgsTestIds):
            anotsTest.append(anot)
        else:
            anotsTrain.append(anot)

    jsonTest['annotations'] = anotsTest
    jsonTrain['annotations'] = anotsTrain

    #json_raws = data_file.readlines()

    with open(r'D:\repos\Labelbox\scripts\train_2.json', 'w') as fp:
        json.dump(jsonTrain, fp)

    with open(r'D:\repos\Labelbox\scripts\validation.json', 'w') as fp:
        json.dump(jsonTest, fp)
