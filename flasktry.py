from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
import cv2                              # Library for image processing
from math import floor

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/shirt.html')
def plot():
    return render_template('shirt.html')
@app.route('/pant.html')
def ploty():
    return render_template('pant.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    shirtno = int(request.form["shirt"])
    pantno = int(request.form["pant"])

    cv2.cv2.waitKey(1)
    cap=cv2.cv2.VideoCapture(0)
    ih=shirtno
    i=pantno
    while True:
        imgarr=["shirt1.png",'shirt2.png','shirt51.jpg','shirt6.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cv2.cvtColor(imgshirt,cv2.cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.cv2.threshold(shirtgray,200 , 255, cv2.cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cv2.cvtColor(imgshirt,cv2.cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.cv2.threshold(shirtgray,0 , 255, cv2.cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        imgarr=["pant7.jpg",'pant21.png']
        #i=input("Enter the pant number you want to try")
        imgpant = cv2.cv2.imread(imgarr[i-1],1)
        imgpant=imgpant[:,:,0:3]#original img in bgr
        pantgray = cv2.cv2.cvtColor(imgpant,cv2.cv2.COLOR_BGR2GRAY) #grayscale conversion
        if i==1:
            ret, orig_mask = cv2.cv2.threshold(pantgray,100 , 255, cv2.cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_mask_inv = cv2.cv2.bitwise_not(orig_mask)
        else:
            ret, orig_mask = cv2.cv2.threshold(pantgray,50 , 255, cv2.cv2.THRESH_BINARY)
            orig_mask_inv = cv2.cv2.bitwise_not(orig_mask)
        origpantHeight, origpantWidth = imgpant.shape[:2]
        face_cascade=cv2.cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.cv2.namedWindow("img",cv2.cv2.WINDOW_NORMAL)
        # cv2.cv2.setWindowProperty('img',cv2.cv2.WND_PROP_FULLSCREEN,cv2.cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
            pantWidth =  3 * w  #approx wrt face width
            pantHeight = pantWidth * origpantHeight / origpantWidth #preserving aspect ratio of original image..

            # Center the pant..just random calculations..
            if i==1:
                x1 = x-w
                x2 =x1+3*w
                y1 = y+5*h
                y2 = y+h*10
            elif i==2:
                x1 = x-w/2
                x2 =x1+2*w
                y1 = y+4*h
                y2 = y+h*9
            else :
                x1 = x-w/2
                x2 =x1+5*w/2
                y1 = y+5*h
                y2 = y+h*14
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            #two cases:
            """
            close to camera: image will be to big
            so face ke x+w ke niche hona chahiye warna dont render at all
            """
            if x1 < 0:
                x1 = 0 #top left ke bahar
            if x2 > img.shape[1]:
                x2 =img.shape[1] #bottom right ke bahar
            if y2 > img.shape[0] :
                y2 =img.shape[0] #nichese bahar
            if y1 > img.shape[0] :
                y1 =img.shape[0] #nichese bahar
            if y1==y2:
                y1=0
            temp=0
            if y1>y2:
                temp=y1
                y1=y2
                y2=temp
            """
            if y+h > y1: #agar face ka bottom most coordinate pant ke top ke niche hai
                y1 = 0
                y2 = 0
            """
            # Re-calculate the width and height of the pant image(to resize the image when it wud be pasted)
            pantWidth = int(abs(x2 - x1))
            pantHeight = int(abs(y2 - y1))
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            #cv2.cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            # Re-size the original image and the masks to the pant sizes
            """
            if not y1 == 0 and y2 == 0:
                pant = cv2.cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA)
                mask_inv = cv2.cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA)
            # take ROI for pant from background equal to size of pant image
                roi = img[y1:y2, x1:x2]
                    # roi_bg contains the original image only where the pant is not
                    # in the region that is the size of the pant.
                num=roi
                roi_bg = cv2.cv2.bitwise_and(roi,num,mask = mask_inv)
                    # roi_fg contains the image of the pant only where the pant is
                roi_fg = cv2.cv2.bitwise_and(pant,pant,mask = mask)
                # join the roi_bg and roi_fg
                dst = cv2.cv2.add(roi_bg,roi_fg)
                    # place the joined image, saved to dst back over the original image
                img[y1:y2, x1:x2] = dst
            """
            
            pant = cv2.cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA)
            mask_inv = cv2.cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.cv2.INTER_AREA)
        # take ROI for pant from background equal to size of pant image
            roi = img[y1:y2, x1:x2]
                # roi_bg contains the original image only where the pant is not
                # in the region that is the size of the pant.
            num=roi
            roi_bg = cv2.cv2.bitwise_and(roi,num,mask = mask_inv)
                # roi_fg contains the image of the pant only where the pant is
            roi_fg = cv2.cv2.bitwise_and(pant,pant,mask = mask)
            # join the roi_bg and roi_fg
            dst = cv2.cv2.add(roi_bg,roi_fg)
                # place the joined image, saved to dst back over the original image
            top=img[0:y,0:resizewidth]
            bottom=img[y+h:resizeheight,0:resizewidth]
            midleft=img[y:y+h,0:x]
            midright=img[y:y+h,x+w:resizewidth]
            blurvalue=5
            top=cv2.GaussianBlur(top,(blurvalue,blurvalue),0)
            bottom=cv2.GaussianBlur(bottom,(blurvalue,blurvalue),0)
            midright=cv2.GaussianBlur(midright,(blurvalue,blurvalue),0)
            midleft=cv2.GaussianBlur(midleft,(blurvalue,blurvalue),0)
            img[0:y,0:resizewidth]=top
            img[y+h:resizeheight,0:resizewidth]=bottom
            img[y:y+h,0:x]=midleft
            img[y:y+h,x+w:resizewidth]=midright
            img[y1:y2, x1:x2] = dst

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-w
            x2s =x1s+3*w
            y1s = y+h
            y2s = y1s+h*4
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
            masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            
            break
        cv2.cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.cv2.destroyAllWindows()                 # Destroys all the windows created by imshow

    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
