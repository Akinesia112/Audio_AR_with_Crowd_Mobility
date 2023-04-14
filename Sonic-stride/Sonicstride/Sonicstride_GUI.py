import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QStackedWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


# Initialize DeepSort
config_file = 'yolov4.cfg'
weights_file = 'yolov4.weights'
# config_file = 'yolov5.cfg'
# weights_file = 'yolov5l.pt'

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




class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # 設置攝像頭
        self.cap = cv2.VideoCapture(0)

        
        # 初始化 UI
        self.init_ui()

        # 設置計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.points = []
        self.M = None
        
        self.tracking_enabled = False

    def init_ui(self):
        self.stacked_widget = QStackedWidget()

        # Step 1
        self.step1_widget = QWidget()
        step1_layout = QVBoxLayout()

        # 設置顯示框
        self.canvas1 = QLabel(self)
        step1_layout.addWidget(self.canvas1)

        # 設置按鈕
        self.start_monitoring_button = QPushButton('開始監控', self)
        self.start_monitoring_button.clicked.connect(self.start_monitoring)
        step1_layout.addWidget(self.start_monitoring_button)

        self.snapshot_button = QPushButton('拍照', self)
        self.snapshot_button.clicked.connect(self.snapshot)
        step1_layout.addWidget(self.snapshot_button)

        self.cancel_button = QPushButton('取消幾何矯正', self)
        self.cancel_button.clicked.connect(self.cancel)
        step1_layout.addWidget(self.cancel_button)

        # 下一步按鈕
        self.next_button = QPushButton("下一步", self)
        self.next_button.clicked.connect(self.next_step)
        self.next_button.setEnabled(False)  # 預設情況下禁用下一步按鈕
        step1_layout.addWidget(self.next_button)

        self.step1_widget.setLayout(step1_layout)
        self.stacked_widget.addWidget(self.step1_widget)

        # Step 2
        self.step2_widget = QWidget()
        step2_layout = QVBoxLayout()

        # 設置顯示框
        self.canvas2 = QLabel(self)
        step2_layout.addWidget(self.canvas2)

        # 設置按鈕
        self.start_monitoring_button2 = QPushButton('開始監控', self)
        self.start_monitoring_button2.clicked.connect(self.start_monitoring)
        step2_layout.addWidget(self.start_monitoring_button2)

        self.start_processing_button = QPushButton('啟動圖像處理', self)
        self.start_processing_button.clicked.connect(self.start_processing)
        step2_layout.addWidget(self.start_processing_button)

        self.stop_button2 = QPushButton('停止監控', self)
        self.stop_button2.clicked.connect(self.stop_monitoring)
        step2_layout.addWidget(self.stop_button2)

        # 上一步按鈕
        self.previous_button = QPushButton("上一步", self)
        self.previous_button.clicked.connect(self.previous_step)
        step2_layout.addWidget(self.previous_button)

        self.step2_widget.setLayout(step2_layout)
        self.stacked_widget.addWidget(self.step2_widget)

        self.setCentralWidget(self.stacked_widget)
        self.setWindowTitle('Webcam Monitor')

    def start_monitoring(self):
        self.timer.start(10)

    def stop_monitoring(self):
        self.timer.stop()

    def update_frame(self):
        ret, self.frame = self.cap.read() 
        if not ret:
            return

        



        

        

        if self.stacked_widget.currentIndex() == 0:
            if self.M is not None:
                self.frame = cv2.warpPerspective(self.frame, self.M, (self.frame.shape[1], self.frame.shape[0]))
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            self.canvas1.setPixmap(pixmap)
        else:
            if self.tracking_enabled:
                self.frame = self.process_video.start_process(ret, self.frame)
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
            self.canvas2.setPixmap(pixmap)


    def previous_step(self):
        if self.stacked_widget.currentIndex() == 1:
            self.stacked_widget.setCurrentIndex(0)
        elif self.stacked_widget.currentIndex() == 0:
            self.stacked_widget.setCurrentIndex(1)


    def next_step(self):
        if self.stacked_widget.currentIndex() == 0:
            self.stacked_widget.setCurrentIndex(1)
        elif self.stacked_widget.currentIndex() == 1:
            self.stacked_widget.setCurrentIndex(0)

    def snapshot(self):
        self.stop_monitoring()
        frame = self.frame
        if self.stacked_widget.currentIndex() == 0:
            if self.cap.isOpened():


                self.points = []
                
                self.next_button.setEnabled(True)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.click_event)
                cv2.imshow('image', gray)
                cv2.waitKey(0)
                cv2.destroyWindow('image')

                if len(self.points) == 4:
                    A, B, C, D = self.points[:4]
                    pts1 = np.float32([A, D, B, C])
                    pts2 = np.float32([[300, 300], [600, 300], [300, 600], [600, 600]])
                    self.M = cv2.getPerspectiveTransform(pts1, pts2)
                    QMessageBox.information(self, '訊息', f'幾何矯正矩陣 = {self.M}')
                    self.start_monitoring()
                else:
                    QMessageBox.warning(self, '警告', '請選擇四個點進行矯正')
                    self.start_monitoring()

    def cancel(self):
        self.points = []
        self.M = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))
            if len(self.points) >= 2:
                cv2.line(self.frame, self.points[-1], self.points[-2], (0, 255, 0), 1)
            cv2.imshow('image', self.frame)

    def start_processing(self):
        if self.M is None:
            QMessageBox.warning(self, '警告', '請先完成幾何矯正')
        else:
            self.process_video = ProcessVideo(self.cap,self.M)

            self.tracking_enabled = True
            self.start_processing_button.setStyleSheet("background-color: red")
            self.start_processing_button.setText('行人追蹤中')
            self.start_processing_button.clicked.disconnect()
            self.start_processing_button.clicked.connect(self.stop_processing)
            


    def stop_processing(self):
        self.tracking_enabled = False
        self.start_processing_button.setStyleSheet("")
        self.start_processing_button.setText('啟動圖像處理')
        self.start_processing_button.clicked.disconnect()
        self.start_processing_button.clicked.connect(self.start_processing)


    def closeEvent(self, event):
        self.cap.release()




class ProcessVideo():
    def __init__(self, cap, M):
        self.cap = cap
        self.M = M
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_count = 0
        self.previous_locations = {}
        self.speeds = {}

        self.fps = 10
        self.k = 10  # 幾張照片平均算一次速度~ k = 走路的週期(1s)*fps


        while self.frame_count < self.k + 1:
            ret, frame = self.cap.read()
            self.start_process(ret, frame)


    def process_image(self, image):
        
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
            if confidence > 0.3 and class_id == 0:
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


    def start_process(self, ret, image):
            if self.frame_count < self.k + 1:
                print('initialize %d %%' % (100 / (self.k + 1) * (self.frame_count)))
            else:
                print('--------- %05d ---------' % (self.frame_count - self.k))

            people_list = self.process_image(image)

            if not people_list:
                print('no people detection')

            total_people_count = 0
            total_speed_sum = 0

            for people in people_list:
                try:
                    
                    x1, y1, x2, y2 = people.original_ltwh
                    bottom_x, bottom_y = (x1+x2)/2, y2

                    # 進行幾何矯正
                    x, y, z = self.M.dot(([bottom_x,bottom_y,1]))
                    bottom_x = x/z
                    bottom_y = y/z

                    track_id = people.track_id

                    if track_id not in self.previous_locations:
                        self.previous_locations[track_id] = []
                        self.previous_locations[track_id].append([bottom_x, bottom_y])

                    else:
                        if len(self.previous_locations[track_id]) > self.k:
                            length = np.sqrt(
                                (bottom_x - self.previous_locations[track_id][-self.k][0]) ** 2 + (
                                            bottom_y - self.previous_locations[track_id][-self.k][1]) ** 2)
                            avg_speed = length / self.k * self.fps  # arbitrary length/sec

                            # save speed
                            if track_id not in self.speeds:
                                self.speeds[track_id] = []
                            self.speeds[track_id].append(avg_speed)

                            print(f"ID {track_id} speed：{avg_speed:.2f}")
                        self.previous_locations[track_id].append((bottom_x, bottom_y))

                        total_speed_sum += avg_speed
                    total_people_count += 1
                except Exception as e:
                    pass

            for people in people_list:
                try:
                    x1, y1, x2, y2 = people.original_ltwh
                    cv2.rectangle(image, tuple([np.int16(x1), np.int16(y1)]), tuple([np.int16(x2), np.int16(y2)]), (0, 255, 0), 2)
                    track_id = people.track_id

                    if track_id in self.speeds:
                        speed = self.speeds[track_id][-1]
                        total_speed_sum += speed
                    else:
                        speed = 0
                    
                    total_people_count += 1
                    
                    cv2.putText(image, f"ID: {track_id}, Speed: {speed:.2f}", tuple([np.int16(x1), np.int16(y1) - 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    pass

            avg_speed = total_speed_sum / total_people_count if total_people_count > 0 else 0
            print(f"Total People: {total_people_count}, Avg Speed: {avg_speed:.2f}")
            cv2.putText(image, f"Total People: {total_people_count}, Avg Speed: {avg_speed:.2f}", tuple([self.width - 600, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



            self.frame_count += 1
            target_time = datetime.datetime.now() + datetime.timedelta(seconds=1 / self.fps)

            while datetime.datetime.now() < target_time:
                time.sleep(0.01)  # Sleep for 0.1 seconds
            
            
            return image




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

