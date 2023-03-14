from AIDetector_pytorch import Detector
import imutils
import cv2
import os
from collections import defaultdict
import tracker as tracker
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def main():

    name = 'demo'
    dict_box=dict()
    
    global frames
    #cap = cv2.VideoCapture('E:/视频/行人监控/test01.mp4')
    cap = cv2.VideoCapture('test.mp4')
    frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    det = Detector()
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    ##
    
    



    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
        if frames>2:    
            for value in dict_box.items():
                for a in range(len(value)-1):
                    color=[0,0, 255]
                    index_start=a                    
                    index_end=index_start+1
                    cv2.line(result,tuple(map(int(value[index_start]),tuple(map(int(value[index_end])))),color,thickness=5))  

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()