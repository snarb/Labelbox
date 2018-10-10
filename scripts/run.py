# import labelbox2coco library
import labelbox2coco as lb2co

# set labeled_data to the file path of the Labelbox JSON export
labeled_data = '1010.json' #'950jsonWTK.json'

# set coco_output to the file name you want the COCO data to be written to
coco_output = 'train.json'


def Replace(path):
    replacements = {'Feldschlösschen':'Feldschloesschen'}

    with open(path, 'r+') as outfile:
        newText = outfile.read()
        for src, target in replacements.items():
            newText = newText.replace(src, target)
        outfile.write(newText)

#Replace(labeled_data)
# call the Labelbox to COCO conversion
lb2co.from_json(labeled_data=labeled_data, coco_output=coco_output)