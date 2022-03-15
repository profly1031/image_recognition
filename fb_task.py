#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import numpy as np

class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0}
        
    def analysis(self, frame):
        
        Z = frame.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        K = 5
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((frame.shape))

        gray = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        #binary = np.where(gray!=67, 255, 0).reshape(gray.shape)
        binary = np.where((gray.flatten()>70) | (gray.flatten()<60), 255, 0).reshape(gray.shape)
        binary = binary.astype(np.uint8)
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        vol = 0
        l_lst = []
        v_lst = []
        for cnt in range(len(contours)):
            # 提取與繪制輪廓
            cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)

            # 輪廓逼近
            epsilon = 0.1 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # 分析幾何形狀
            corners = len(approx)
            self.shapes['triangle'] += 1
            mm = cv.moments(contours[cnt])

            cx = int(mm['m10'] / (mm['m00']+1))
            cy = int(mm['m01'] / (mm['m00']+1))
            if corners == 3:
                cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv.putText(frame, str(cnt), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2, cv.LINE_AA)
                p = cv.arcLength(contours[cnt], True)
                area = cv.contourArea(contours[cnt])
                print("第%.0f個圖形:\"三角形\",邊長: %.3f, 面積: %.3f"% (cnt ,p/3, area))
                vol += area
                
                l_lst.append(p/3)
                v_lst.append(area)
            else:
                cv.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                cv.putText(frame, str(cnt), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                p = cv.arcLength(contours[cnt], True)
                area = cv.contourArea(contours[cnt])
                print("!!第%.0f個圖形:\"非三角形\",周長: %.3f, 面積: %.3f"% (cnt ,p, area))
        total_v = vol/(frame.shape[0]*frame.shape[1])
        print("-----------------------")
        print('三角形總面積比例: %.3f'%(total_v) )
        cv.namedWindow('Analysis Result', 0)
        cv.imshow("Analysis Result", frame)
        #cv.imwrite("D:/test-result.png", self.draw_text_info(result))
        return l_lst, v_lst, total_v


if __name__ == "__main__":
    src = cv.imread("fb1.png")
    ld = ShapeAnalysis()
    #以下分別為，邊長的list，面積的list,總面積的scalar
    l, v, vt = ld.analysis(src)
    #擷取範例，假設預找第5個三角形的資訊
    #n = 4
    #print("第%.0f個三角形:邊長%.3f,面積%.3f"%(n, l[n],v[n]))
    cv.waitKey(0)
    cv.destroyAllWindows()

