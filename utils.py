import cv2 as cv
import numpy as np


def get_contours(img, can_threshold=[100, 100], show_canny=False, min_area=1000, filter=0, draw=False):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv.Canny(img_blur, can_threshold[0], can_threshold[1])
    kernel = np.ones((5, 5))
    img_dial = cv.dilate(img_canny, kernel, iterations=3)
    img_threshold = cv.erode(img_dial, kernel, iterations=2)

    if show_canny:
        cv.imshow("Canny", img_threshold)

    contours, hiearchy = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for i in contours:
        area = cv.contourArea(i)
        if area > min_area:
            perimeter = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * perimeter, True)
            b_box = cv.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    final_contours.append([len(approx), area, approx, b_box])
            else:
                final_contours.append([len(approx), area, approx, b_box])

    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in final_contours:
            cv.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, final_contours


def reorder(my_points):
    print(my_points.shape)
    my_new_points = np.zeros_like(my_points)
    my_points = my_points.reshape((4, 2))
    add = my_points.sum(1)
    my_new_points[0] = my_points[np.argmin(add)]
    my_new_points[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_new_points[1] = my_points[np.argmin(diff)]
    my_new_points[1] = my_points[np.argmax(diff)]
    return my_new_points


def warp_img(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_warp = cv.warpPerspective(img, matrix, (w, h))
    img_warp = img_warp[pad:img_warp.shape[0] - pad, pad:img_warp.shape[1]-pad]
    return img_warp

def find_distance(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5



# a = get_contours('2.jpg', True)
