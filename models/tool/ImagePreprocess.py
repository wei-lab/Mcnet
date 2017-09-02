import cv2
import numpy as np
import random
import os

def resize_img(fold,img_path,w):

    img=cv2.imread(img_path)
    fname=img_path.split('/')[-1]
    width=img.shape[1]
    height=img.shape[0]
    print (width,height)
    img_height=int(height*(w/float(width)))
    seed=random.randint(0,1)
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    img_new=cv2.resize(img,(w,img_height))
    if(seed==0):
        img_dis=cv2.copyMakeBorder(img_new,w-img_height,0,0,0, cv2.BORDER_CONSTANT,
                                       value=(r, g, b))
    else:
        img_dis=cv2.copyMakeBorder(img_new,0,w-img_height,0,0, cv2.BORDER_CONSTANT,
                                       value=(r, g, b))
    cv2.imwrite(fold+"/new_"+fname,img_dis)


def random_briteness(img_path,fold):
    img=cv2.imread(img_path)
    fname=img_path.split('/')[-1]
    width=img.shape[1]
    height=img.shape[0]
    for i in range(3):
        seed=random.randint(8,12)/float(10)
        for xi in range(width):
            for xj in range(height):

                img[xj,xi,0]=int(img[xj,xi,0]*seed)
                img[xj,xi,1]=int(img[xj,xi,1]*seed)
                img[xj,xi,2]=int(img[xj,xi,2]*seed)
       # img2= cv2.resize(img, (200, 200))
       #  cv2.namedWindow('img')
       #  cv2.imshow('img',img)
       #  cv2.waitKey()
        cv2.imwrite(fold+"/new_brightness_"+str(i+1)+fname,img)

def rotate_img(img_path,fold):
    img=cv2.imread(img_path)
    fname=img_path.split('/')[-1]
    seed=random.randint(1,10)
    theta =seed * np.pi / 180
    M_rotate = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0]
        ], dtype=np.float32)
    img_rotated = cv2.warpAffine(img, M_rotate, (128, 128))

    # cv2.namedWindow('img')
    # cv2.imshow('img',img_rotated )
    # cv2.waitKey()

    cv2.imwrite(fold+"/new_rotate_"+fname,img_rotated)

if __name__=='__main__':
    fold='../Testimg_2N'

    parhDir=os.listdir("../2N")

    for allDir in parhDir:

        child = os.path.join('%s/%s' % ('../2N', allDir))
        #rotate_img(child,fold)

        resize_img(fold,child,128)
        #random_briteness(child,fold)


    #random_briteness('/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/2N/3.JPG')
    # #resize_img('/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/2N',
    #            '/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/2N/3.JPG',
    #            128)
