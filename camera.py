import cv2
from os import listdir, remove, path, rmdir
import re
from detector import Detector
from glob import glob


class Camera:
    def __init__(self, video_source=0, pet=False):
        self.cap = cv2.VideoCapture(video_source)
        self.is_pet = pet
        self.detector = Detector()
        self.consecutive = True
        self.frame_dir = "frames"

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def save_detected_motion(self, img):

        frame_list = listdir(self.frame_dir)
        if frame_list:
            frame_list.sort(key=self.natural_keys)
            frame_number = re.findall(r'\d+', frame_list[-1])[0]
            cv2.imwrite(f"frames/frame#{int(frame_number) + 1}.png", img)
        else:
            cv2.imwrite("frames/frame#1.png", img)

    def make_video(self):
        img_array = []
        for filename in glob('frames/*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        try:
            video_list = listdir("motions")
            if video_list:
                video_list.sort(key=self.natural_keys)
                video_number = int(re.findall(r'\d+', video_list[-1])[0])
            else:
                video_number = 0
            out = cv2.VideoWriter(f'motions/motion#{video_number + 1}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
            for image in img_array:
                out.write(image)
            out.release()
            for f in listdir(self.frame_dir):
                remove(path.join(self.frame_dir, f))
            rmdir(self.frame_dir)
        except NameError:
            return

    def run(self):
        while True:
            _, img = self.cap.read()
            if self.detector.detect(img, self.is_pet):
                self.save_detected_motion(img)
                self.consecutive = True
            else:
                self.consecutive = False
            if not self.consecutive:
                self.make_video()
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
