from depthai_sdk import OakCamera, TwoStagePacket
import numpy as np
import cv2
import os

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
# Сопоставление эмоций и эмодзи
emotion_emojis = {'neutral': 'neutral1.png', 'happy': 'smile1.png', 'sad': 'sad1.png',
                  'surprise': 'surprise1.png', 'anger': 'angry1.png'}

def overlay_emoji(frame, top_left, bottom_right, emotion):
    frame_height, frame_width = frame.shape[:2]

    # Ensure that top_left and bottom_right are within the frame
    if (top_left[0] < 0 or top_left[1] < 0 or
        bottom_right[0] > frame_width or bottom_right[1] > frame_height):
        print("Face detection is out of frame boundaries. Skipping overlay.")
        return

    emoji_path = emotion_emojis.get(emotion)
    if not emoji_path or not os.path.exists(emoji_path):
        print(f'Emoji file for {emotion} not found')
        return
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

    # Convert emoji colors from BGRA to RGBA
    emoji = cv2.cvtColor(emoji, cv2.COLOR_BGRA2RGBA)

    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    if width <= 0 or height <= 0:
        print(f'Invalid dimensions: width {width}, height {height}')
        return

    emoji = cv2.resize(emoji, (width, height))

    # Extract color and alpha channels
    emoji_bgr = emoji[:,:,:3]
    emoji_mask = emoji[:,:,3]

    # Mask for the frame area where emoji will be overlaid
    frame_mask = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Apply emoji on the frame
    bg = cv2.bitwise_and(frame_mask[:,:,:3], frame_mask[:,:,:3], mask=cv2.bitwise_not(emoji_mask))
    fg = cv2.bitwise_and(emoji_bgr, emoji_bgr, mask=emoji_mask)

    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.add(bg, fg)

with OakCamera() as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', color)
    det.config_nn(resize_mode='crop')

    emotion_nn = oak.create_nn('emotions-recognition-retail-0003', input=det)

    def cb(packet: TwoStagePacket, visualizer):
        for det, rec in zip(packet.detections, packet.nnData):
            emotion_results = np.array(rec.getFirstLayerFp16())
            emotion_name = emotions[np.argmax(emotion_results)]
            overlay_emoji(packet.frame, det.top_left, det.bottom_right, emotion_name)

            visualizer.draw(packet.frame)
            cv2.imshow(packet.name, packet.frame)

    oak.visualize(emotion_nn, callback=cb, fps=True)
    oak.visualize(det.out.passthrough)
    oak.start(blocking=True)
