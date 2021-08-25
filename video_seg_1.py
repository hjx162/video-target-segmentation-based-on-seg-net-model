# -*- coding=GBK -*-
import cv2 as cv
import os
import tensorflow as tf
import numpy as np


video_path = os.path.join('E:/Python/pythonProject_4/target_tracking_and_detection/cat_1.mp4') # or dog.mp4
times=0
#提取视频的频率，每25帧提取一个
frameFrequency=1
#输出图片到当前目录vedio文件夹下
outPutDirName='E:/Python/pythonProject_4/target_tracking_and_detection/result_seg_save/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv.VideoCapture(video_path)

cv.namedWindow("seg_video", 0)

while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        # cv.imwrite(outPutDirName + str(times)+'.png', image)
        # print(outPutDirName + str(times) + '.png')
        image_1 = image.astype("float32")  # 数据类型转换 : 将读入图片的int8类型转换为float32类型
        image_1 = image_1 / 255
        image_1 = cv.resize(image_1, (224, 224))
        image_2 = np.expand_dims(image_1, axis=0)
        image_2 = tf.convert_to_tensor(image_2)

        pred_mask = model.predict(image_2)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]

        pred_mask_1 = tf.keras.preprocessing.image.array_to_img(pred_mask[0]) #PIL转
        pred_mask_2 = np.asarray(pred_mask_1)
        depth_image = cv.applyColorMap(pred_mask_2, cv.COLORMAP_VIRIDIS)
        depth_image = depth_image.astype("float32")
        depth_image = depth_image / 255

        imgs = np.hstack((image_1,depth_image))
        cv.imshow("seg_video",imgs)

        cv.imwrite(outPutDirName + str(times)+'.png', depth_image*255)
        print(outPutDirName + str(times) + '.png')


print('图片提取结束')
camera.release()
cv.destroyAllWindows()