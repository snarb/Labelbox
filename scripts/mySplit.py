import json
import random
import numpy as np
from random import shuffle
import copy

testRate = 0.15


with open(r'D:\fakeCoco\train.json', encoding='utf-8') as data_file:

    jsonObject = json.loads(data_file.read())

    jsonTest = copy.deepcopy(jsonObject)
    jsonTrain = copy.deepcopy(jsonObject)

    anots = jsonObject['annotations']
    shuffle(anots)
    testSize = int(testRate * len(anots))

    anotsTest = anots[:testSize]
    anotsTrain = anots[testSize:]

    jsonTest['annotations'] = anotsTest
    jsonTrain['annotations'] = anotsTrain

    #json_raws = data_file.readlines()

    with open(r'D:\fakeCoco\train_2.json', 'w') as fp:
        json.dump(jsonTrain, fp)

    with open(r'D:\fakeCoco\validation.json', 'w') as fp:
        json.dump(jsonTest, fp)
