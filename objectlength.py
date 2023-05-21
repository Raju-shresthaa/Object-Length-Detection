import cv2 as cv
import numpy as np
import utils

webcam = False
path = '2.jpg'
cap = cv.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 4032)
cap.set(4, 3024)
scale = 3
wp = 210*scale
hp = 297*scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv.imread('2.jpg')
    img_contours, conts = utils.get_contours(img, min_area=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        img_warp = utils.warp_img(img, biggest, wp, hp)
        img_contours2, conts2 = utils.get_contours(img_warp, min_area=50000, filter=4,can_threshold=[50,50],draw=True)
        if len(conts) != 0:
            for obj in conts:
                cv.polylines(img_contours2, [obj[2]],True,(0,255,0),2)
                n_points = utils.reorder(obj[2])
                n_w = round((utils.find_distance(n_points[0][0]//scale, n_points[1][0]//scale)),1)
                n_h = round((utils.find_distance(n_points[0][0]//scale, n_points[2][0]//scale)),1)
                cv.arrowedLine(img_contours2, (n_points[0][0][0], n_points[0][0][1]),
                                (n_points[1][0][0], n_points[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv.arrowedLine(img_contours2, (n_points[0][0][0], n_points[0][0][1]),
                                (n_points[2][0][0], n_points[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv.putText(img_contours2, '{}cm'.format(n_w), (x + 30, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv.putText(img_contours2, '{}cm'.format(n_h), (x - 70, y + h // 2), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)


        cv.imshow('A4', img_contours2)


    img = cv.resize(img, (0, 0), None, 0.5, 0.5)
    cv.imshow('Original', img)
    cv.waitKey(1)
