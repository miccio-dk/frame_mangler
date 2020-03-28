import os
import time
from collections import deque

import cv2
import numpy as np


## class for processing frames
class FrameProcessor():
    font = cv2.FONT_HERSHEY_PLAIN

    # constructor
    def __init__(self, buff_max_len, jitter_threshold, feedback_threshold, mode, resize=None, show_infos=False):
        self.buffer = deque()
        self.t_start = 0 
        self.t_end = 0
        self.buff_max_len = buff_max_len
        self.jitter_threshold = jitter_threshold
        self.feedback_threshold = feedback_threshold
        self.mode = mode
        self.resize = resize
        self.show_infos = show_infos

    # append frame channels to buffer
    def append_frame(self, frame):
        self.buffer.append(frame[:,:,0])
        self.buffer.append(frame[:,:,1])
        self.buffer.append(frame[:,:,2])

    # retrieve frame channels from buffer
    def get_frame(self, idxs=None):
        if idxs is not None:
            a = self.buffer[idxs[0]]
            b = self.buffer[idxs[1]]
            c = self.buffer[idxs[2]]
            self.buffer.popleft()
            self.buffer.popleft()
            self.buffer.popleft()
        else:
            a = self.buffer.popleft()
            b = self.buffer.popleft()
            c = self.buffer.popleft()
        return np.stack([a, b, c], axis=2)

    # process frame: take a new frame and return one from buffer
    def process(self, img_input):
        if self.resize is not None:
            in_shape = img_input.shape
            out_size = int(in_shape[1] * self.resize), int(in_shape[0] * self.resize)
            img_input = cv2.resize(img_input, out_size, interpolation=cv2.INTER_AREA)
            print(out_size)

        # if mode is HSV, decompose img
        if self.mode in ['hsv', 'hsv2']:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
        
        # if buffer is right size, apply jitter
        if len(self.buffer) == self.buff_max_len:
            if np.random.rand() < self.jitter_threshold:
                # depending on mode, choose which frame channels to return
                if self.mode == 'std':
                    idx = np.random.choice(np.arange(self.buff_max_len)) // 3 * 3
                    idxs = np.array([idx, idx+1, idx+2])
                elif self.mode == 'rgb':
                    idxs = np.random.choice(np.arange(self.buff_max_len), 3)
                elif self.mode == 'hsv':
                    idx1 = np.random.choice(np.arange(self.buff_max_len)) // 3 * 3
                    idx2 = np.random.choice(np.arange(self.buff_max_len)) // 3 * 3
                    idx3 = np.random.choice(np.arange(self.buff_max_len)) // 3 * 3
                    idxs = np.array([idx1, idx2+1, idx3+2])
                elif self.mode == 'hsv2':
                    idxs = np.random.choice(np.arange(self.buff_max_len), 3)
                # return whole frame from buffer
                img_output = self.get_frame(idxs)
                
            else:
                img_output = self.get_frame()

        # if buffer is full, discard one frame
        elif len(self.buffer) > self.buff_max_len:
            self.buffer.popleft()
            self.buffer.popleft()
            self.buffer.popleft()
            img_output = self.get_frame()
        
        else:
            img_output = img_input

        # if mode is HSV, recompose img
        if self.mode in ['hsv', 'hsv2']:
            img_output = cv2.cvtColor(img_output, cv2.COLOR_HSV2BGR)

        # apply feedback
        if np.random.rand() < self.feedback_threshold:
            self.append_frame(img_output)
        else:
            self.append_frame(img_input)

        # show debug infos on screen
        img_output_ret = img_output.copy()
        if self.show_infos:
            # calculate FPS
            self.t_end = time.time()
            fps = 1 / (self.t_end - self.t_start)
            self.t_start = self.t_end
            # calculate n frames
            n_frames = len(self.buffer) // 3
            n_frames_s = n_frames / fps
            cv2.putText(img_output_ret, f'Buffer length: {n_frames} frames ({n_frames_s:.1f}s)', (10, 520), FrameProcessor.font, 1, (255,255,255), 1)
            cv2.putText(img_output_ret, f'Jitter: {self.jitter_threshold:.2f}',      (10, 500), FrameProcessor.font, 1, (255,255,255), 1)
            cv2.putText(img_output_ret, f'Mode: {self.mode}',                        (10, 540), FrameProcessor.font, 1, (255,255,255), 1)
            cv2.putText(img_output_ret, f'Feedback: {self.feedback_threshold:.2f}',  (10, 560), FrameProcessor.font, 1, (255,255,255), 1)
            cv2.putText(img_output_ret, f'FPS: {fps:.2f}',                           (900, 560), FrameProcessor.font, 1, (255,255,255), 1)

        return img_output_ret

    # callbacks for parameter change
    def bufflen_onchange_cb(self, val):
        self.buff_max_len = int(2 + val // 4) * 3

    def jitter_onchange_cb(self, val):
        self.jitter_threshold = val / 100

    def feedback_onchange_cb(self, val):
        self.feedback_threshold = val / 100

    def mode_onchange_cb(self, val):
        self.mode = ['std', 'rgb', 'hsv', 'hsv2'][val]

    def mouse_cb(self, event, x, y, flags, param):
        pass


## main function
def main():
    # camera and frame processor obj
    cam = cv2.VideoCapture(0)
    fp = FrameProcessor(buff_max_len=30, 
                        jitter_threshold=0.0, 
                        feedback_threshold=0.0, 
                        mode='std', 
                        resize=0.8, 
                        show_infos=True)
    # window
    windowName = 'window'
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, fp.mouse_cb)
    # controls
    cv2.createTrackbar('Jitter', windowName, 0, 100, fp.jitter_onchange_cb)
    cv2.createTrackbar('Buffer length', windowName, 0, 100, fp.bufflen_onchange_cb)
    cv2.createTrackbar('Mode', windowName, 0, 3, fp.mode_onchange_cb)
    cv2.createTrackbar('Feedback', windowName, 0, 100, fp.feedback_onchange_cb)
    # main loop
    while True:
        # acquire, process, and show frame
        _, img_input = cam.read()
        img_output = fp.process(img_input)
        cv2.imshow(windowName, img_output)
        # quit on ESC or Q
        if cv2.waitKey(1) & 0xFF in [27, 113]:
            break
    # quit gracefully
    cam.release()
    cv2.destroyAllWindows()


## ready set go!
if __name__ == '__main__':
    main()