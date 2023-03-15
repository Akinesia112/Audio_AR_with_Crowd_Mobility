import cv2
import numpy as np

#影片導入
vidcap= cv2.VideoCapture("1207video.ts")
success, image = vidcap.read()
count = 0
while success:
    success, image = vidcap.read()

    #讀取影片畫面轉化為numpy array時定義為int32型，而cv2.resize函數參數必須是浮點型
    #所以此函數為將資料結構轉為flaot格式
    img_array = np.array(image).astype("float") 

    #影片尺寸縮放(方便找尋方位座標)
    frame = cv2.resize(image,(640,480))

    #設定變形點
    tl = (20,150)    #左上topleft
    bl = (300,600)    #左下bottomleft
    tr = (300,150)   #右上topright
    br = (640,260)   #右下bottomright

    #給變形點圖形標示
    cv2.circle(frame,tl,5,(0,0,255),-1)
    cv2.circle(frame,bl,5,(0,0,255),-1)
    cv2.circle(frame,tr,5,(0,0,255),-1)
    cv2.circle(frame,br,5,(0,0,255),-1)

    #需導入numpy才能使用np.float32
    #np.float32只能讀取單一數劇所以裡面的list資料都要用[]框起來

    #變形前座標位置
    pts1 = np.float32([tl,bl,tr,br])
    #變形後座標位置
    pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    #透視變換的矩陣函數
    #取得變形數據
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    #設定變形參數 (畫面來源,數據來源,畫面大小)
    transformed_frame = cv2.warpPerspective(frame,matrix,(640,480))


    cv2.imshow("Frame",frame)
    cv2.imshow("transformed_frame Kenting plan view",transformed_frame)

    if cv2.waitKey(10) ==27:
        break