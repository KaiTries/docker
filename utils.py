import io
import base64
from PIL import Image
import numpy as np
import cv2
import math
from scipy.ndimage.measurements import center_of_mass
from tensorflow import keras
#example b64 encoded image received through API

class ImageHandler:

    #accept image from API and decode it
    def retrieveB64(postRequest):
        image = base64.b64decode(postRequest,validate=True)
        decoded_string = io.BytesIO(image)
        img = Image.open(decoded_string)
        return img


    #convert Image to array for model
    def saveImage(image):
        image = image.convert("L")
        image = image.resize((28,28))
        image.save("t.png")

    #convert to grayscale and nparray
    def ImageForModel(image):
        image = image.convert("L")
        array = np.array(image) 
        return array
    

    def getBestShift(img):
        cy,cx = center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty

    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted

    def rec_digit(Image_array):
        gray = Image_array

        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)
        while np.sum(gray[-1]) == 0:    
            gray = gray[:-1]  
        while np.sum(gray[:,-1]) == 0:  
            gray = np.delete(gray,-1,1)   
        rows,cols = gray.shape  

        if rows > cols: 
            factor = 20.0/rows    
            rows = 20 
            cols = int(round(cols*factor))    
            gray = cv2.resize(gray, (cols,rows))
        else:   
            factor = 20.0/cols    
            cols = 20 
            rows = int(round(rows*factor))    
            gray = cv2.resize(gray, (cols, rows))     

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor(    (28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor(    (28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')        

        shiftx,shifty = ImageHandler.getBestShift(gray)
        shifted = ImageHandler.shift(gray,shiftx,shifty)
        gray = shifted
        #cv2.imwrite("out.png",gray)
        img = gray / 255.0
        img = np.array(img).reshape(-1, 28, 28, 1)
        return img
    

class ModelHandler:

    def __init__(self):
        self.model = keras.models.load_model('model.h5')

    def predict(self,image):
        prediction = self.model.predict(image).argmax()
        return prediction