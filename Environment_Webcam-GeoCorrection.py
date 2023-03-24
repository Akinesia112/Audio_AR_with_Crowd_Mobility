import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime


cap = cv2.VideoCapture("test.mp4") #選擇編號


##先擷取圖片來做校正設定

time.sleep(3) #倒數三秒

ret, img = cap.read() #拍一張照片


## 點一個透視正方框(從左上開始 順時鐘點)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(255,0,0),-1)
        points.append((x,y))
        if len(points) >= 2:
            cv2.line(img,points[-1],points[-2],(0,255,0),1)
        cv2.imshow("image", img)
        print(points)
cv2.imshow("image",img)
points = []
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)

A=points[0]
B=points[1]
C=points[2]
D=points[3]


'''

intersections = []
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        x3, y3, x4, y4 = lines[j][0]
        
        m = (y1-y2)/(x1-x2)
        n = (y3-y4)/(x3-x4)
        a = y1 - m*x1
        b = y3 - n*x3
        
        if abs(m*n) < 0.4 and  abs(m-n)>0.01:
            px = (b-a)/(m-n)
            py = m*px + a
            if px >0 and py > 0:
                intersections.append([px, py])


# remove duplicates
threshold = 100 
data = np.array(A,B,C,D)

# 建立一個空的布林遮罩
mask = []

# 比較每兩個項是否相近，如果是就將遮罩設為True

for i in range(len(data)):
    for j in range(i+1, len(data)):
        if abs(data[i][0] - data[j][0]) + abs(data[i][1] - data[j][1]) < threshold:
            mask.append(j)

# 使用np.delete()函數刪除遮罩為True的項
intersections = [np.delete(data.T[0], mask) ,np.delete(data.T[1], mask)]

# # sort intersections by clockwise order
center = np.mean(intersections, axis=0)
angles = np.arctan2(intersections[1] - center[1], intersections[0] - center[0])
print(angles)
indices = np.argsort(angles)
intersections[0] = intersections[0][indices]
intersections[1] = intersections[1][indices]


# Display the image

plt.plot(intersections[0],intersections[1],'r.')
print(intersections)
plt.xlim([0,1800])
plt.ylim([1800,0])
plt.show()
'''

# get perspective transform matrix
pts1 = np.float32([A,D,B,C])
pts2 = np.float32([[300, 300], [600, 300],[300, 600],[600, 600]])
M = cv2.getPerspectiveTransform(pts1, pts2) #之後都會用他
print('幾何矯正矩陣 = ', M)


# warp perspective
dst = cv2.warpPerspective(img, M, (1500, 1500))


plt.imshow(dst)
plt.show()

#####################################################################
config_file = 'yolov4.cfg'
weights_file = 'yolov4.weights'

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

deepsort = DeepSort()

def process_image(image):
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    detections = net.forward(output_layers)

    height, width, _ = image.shape
    threshold = 50
    person_detections = []
    person_detections_input = []
    
    for detection in detections[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
            x1 = center_x
            y1 = center_y + h / 2
            mask = 0
            for i in range(len(person_detections)):
                if abs(x1 - person_detections[i][0]) + abs(y1 - person_detections[i][1]) < threshold:
                    mask = 1
                
            if mask == 0:
                person_detections.append([x1, y1])
                person_detections_input.append([[x1 - w/2, y1-h, w, h], confidence])

    embed = deepsort.generate_embeds(image, person_detections_input)
    det = deepsort.create_detections(person_detections_input, embed)
    raw_detections = [[detection.to_tlbr().tolist(),detection.confidence,detection.class_name] for detection in det]

    People_list = deepsort.update_tracks(raw_detections, embed)
    return People_list



# 使用 Webcam 進行影像捕捉
cap = cv2.VideoCapture("test.mp4")

print('webcam ok',cap)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 如果不要特別錄影
output_video_path = 'output_video1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

frame_count = 0
previous_locations = {}
speeds = {}

fps = 5
k = 10 #幾張照片平均算一次速度~ k = 走路的週期(1s)*fps

while True:
    total_people_count = 0
    total_speed_sum = 0
    
    ret, image = cap.read()
    
    target_time = datetime.datetime.now() + datetime.timedelta(seconds=1/fps) # 10 
    
    if not ret:
        break
    if frame_count < k+1:
        print('initialize %d %%' %(100/(k+1)*(frame_count)))
    else:
        print('--------- %05d ---------' %(frame_count-k))
        
    #開始處理影像
    People_list = process_image(image)

    if People_list == []:
        print('no people detection')
    
    
    for people in People_list:
        
        try:
        
            x1, y1, x2, y2 = people.original_ltwh
            bottom_x, bottom_y = (x1+x2)/2, y2

            # 進行幾何矯正
            x, y, z = M.dot(([bottom_x,bottom_y,1]))
            bottom_x = x/z
            bottom_y = y/z

            track_id = people.track_id

            if track_id not in previous_locations:
                previous_locations[track_id] = []
                previous_locations[track_id].append([bottom_x, bottom_y])
            
            else:
                
                if len(previous_locations[track_id]) > k:
                    length = np.sqrt((bottom_x - previous_locations[track_id][-k][0])**2 + (bottom_y - previous_locations[track_id][-k][1])**2)
                    avg_speed = length/k*fps # abitrary length/sec
                    
                    # save speed
                    if track_id not in speeds:
                        speeds[track_id] = []
                    speeds[track_id].append(avg_speed)
                    
                    print(f"ID {track_id} speed：{avg_speed:.2f}")
                previous_locations[track_id].append((bottom_x, bottom_y))  
                
                total_speed_sum += avg_speed
            total_people_count += 1
        except Exception as e:
            1
        #print( bottom_x, bottom_y,people.track_id)
    
    frame_count += 1
    

    # 繪製邊界框、ID 和速度
    for people in People_list:
        try:
            x1, y1, x2, y2 = people.original_ltwh
            cv2.rectangle(image, tuple([np.int16(x1), np.int16(y1)]), tuple([np.int16(x2), np.int16(y2)]), (0, 255, 0), 2)
            track_id = people.track_id

            speed = speeds[track_id][-1] if track_id in speeds else 0  
            cv2.putText(image, f"ID: {track_id}, Speed: {speed:.2f}", tuple([np.int16(x1), np.int16(y1) - 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            1

    # 在右上角顯示總人數和平均速度
    avg_speed = total_speed_sum / total_people_count if total_people_count > 0 else 0
    print(f"Total People: {total_people_count}, Avg Speed: {avg_speed:.2f}")
    cv2.putText(image, f"Total People: {total_people_count}, Avg Speed: {avg_speed:.2f}", tuple([width - 600, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # 將帶有邊界框的圖像寫入影片輸出文件
    out.write(image)
    
    
    while datetime.datetime.now() < target_time:
        time.sleep(0.01) # Sleep for 0.1 seconds
    

    
cap.release()
out.release() # 釋放 VideoWriter 資源
cv2.destroyAllWindows()