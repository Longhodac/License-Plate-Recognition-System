import os
import segmentation
import joblib

# load the model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_path)

classfication_result = []
for each_character in segmentation.characters:
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classfication_result.append(result)
    
print(classfication_result)

plate_string = ''
for eachPredict in classfication_result:
    plate_string += eachPredict[0]

print(plate_string)

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)