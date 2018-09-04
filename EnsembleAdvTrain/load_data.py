import os
import numpy as np
from PIL import Image
from dependency import *
def load_images(path,num_classes):
    #Load images
    
    """print('Loading ' + str(num_classes) + ' classes')

    X_train=np.zeros([num_classes*500,3,64,64],dtype='uint8')
    y_train=np.zeros([num_classes*500], dtype='uint8')
"""
    trainPath=path+'/train'

    print('loading training images...')
   
    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        """for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i]=np.array([X,X,X])
            else:
                X_train[i]=np.transpose(X,(2,0,1))
            y_train[i]=j
            i+=1"""
        j+=1
        if (j >= num_classes):
            break

   # print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([256,3,64,64],dtype='uint8')
    y_test = np.zeros([256], dtype='uint8')


    print('loading test images...')

    i = 0
    testPath=path+'/val/images'
    for sChild in os.listdir(testPath) :
        if i > 255 :
            break
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_test[i]=np.array([X,X,X])
            else:
                X_test[i]=np.transpose(X,(2,0,1))
            y_test[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass


    print('finished loading test images')+str(i)
    #y_train = one_hot_encode(y_train,200)
    y_test = one_hot_encode(y_test,200)
    return np.transpose(X_test,(0,3,2,1)),y_test#np.transpose(X_train,(0,3,2,1)),y_train,

def one_hot_encode(inputs, encoded_size):
    def get_one_hot(number):
        on_hot=[0]*encoded_size
        on_hot[int(number)]=1
        return on_hot
    return np.array(list(map(get_one_hot, inputs)))

def get_annotations_map():
	valAnnotationsPath = '../Defense_Model/tiny-imagenet-200/val/val_annotations.txt'
	valAnnotationsFile = open(valAnnotationsPath, 'r')
	valAnnotationsContents = valAnnotationsFile.read()
	valAnnotations = {}

	for line in valAnnotationsContents.splitlines():
		pieces = line.strip().split()
		valAnnotations[pieces[0]] = pieces[1]

	return valAnnotations

