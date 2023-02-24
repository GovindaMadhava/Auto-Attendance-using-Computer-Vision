from keras.models import load_model
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import os


data=np.load('data.npy')
target=np.load('target.npy')
print(data.shape)
print(target)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('./Test',target_size=(100, 100),batch_size=32,class_mode='categorical',color_mode="grayscale")

data_path='Training'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(labels,categories)) #empty dictionary
print(labels)
print(label_dict)

model = load_model('model-010.model')
model.summary()



ev=model.evaluate(data,target)
print(ev)
result=model.predict(data)

count=0
img=361
imgpoint=0
err=0
curr=0
result = model.predict(data)
for i in result:

    print(f"image number in test directory {img}")
    prev=curr
    actual=np.argmax(target[count])
    curr=actual

    print('Label is : ',label_dict[actual])
    pred=np.argmax(i)
    print('Prediction is: ',label_dict[np.argmax(i)])


    count+=1
    img+=1
    imgpoint+=1
    if  actual !=pred:


        err+=1
        print("incorrect")
    if prev!=curr:

        img=361
        imgpoint=0





print(err)
