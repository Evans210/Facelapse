# Standard library imports
import argparse
import math
import random
import os
import re
import sys
from datetime import datetime

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np


# ---------- Constants ----------
DATE_PATTERN = r"(20\d{2}[01]\d[0-3]\d_[0-2]\d[0-5]\d[0-5]\d)" # YYYYMMDD_HHMMSS
DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 1920
DEFAULT_FPS = 30
DEFAULT_DURATION = 1
DEFAULT_ZOOM = 0.8
DEFAULT_VERTICAL = 0.35
DEFAULT_HORIZONTAL = 0.35


def sort_images(folder, randomize=False):
    images = []
    
    for file in os.listdir(folder):
        if file.lower().endswith('.jpg'):
            images.append(os.path.join(folder, file))
        else:
            print(f"Skipping file: {file}")
    
    if randomize:
        random.shuffle(images)
        return images
    
    entries = []
    
    for image in images:
        match = re.search(DATE_PATTERN, os.path.basename(image))
        if match:
            s = match.group(1)
            try:
                dt = datetime.strptime(s, '%Y%m%d_%H%M%S')
                entries.append((dt, image))
            except ValueError:
                print(f"Error parsing date: {s}")
    
    entries.sort(key=lambda x: x[0])
    return [f for _, f in entries]

class FaceAligner:
    def __init__(self, target_w, target_h, zoom, vertical, horizontal, replicate_border=False):
        self.target_w = target_w
        self.target_h = target_h
        self.zoom = zoom
        self.vertical = vertical
        self.horizontal = horizontal
        self.replicate_border = replicate_border

        # mediapipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_fm = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.5)


    def detect_eyes_mediapipe(self, img: np.ndarray):
        # returns (left_eye_xy, right_eye_xy) in image coords
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_fm.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        left_idxs = [33, 133, 160, 159, 158, 153, 144, 145]
        right_idxs = [362, 263, 387, 386, 385, 380, 373, 374]
        def avg_point(idxs):
            x = np.mean([lm.landmark[i].x for i in idxs]) * w
            y = np.mean([lm.landmark[i].y for i in idxs]) * h
            return (x, y)
        left = avg_point(left_idxs)
        right = avg_point(right_idxs)
        # ensure left is actually left
        if left[0] > right[0]:
            left, right = right, left
        return left, right

    def align_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        eyes = self.detect_eyes_mediapipe(img)

        (lx, ly), (rx, ry) = eyes
        # compute angle
        dx = rx - lx
        dy = ry - ly
        angle = math.degrees(math.atan2(dy, dx))
        # desired distance between eyes in output (as pixels)
        desired_left_x = self.horizontal * self.target_w
        desired_right_x = (1 - self.horizontal) * self.target_w
        desired_dist = desired_right_x - desired_left_x
        current_dist = math.hypot(dx, dy)
        scale = self.zoom * (desired_dist / current_dist)
        # center between eyes
        eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

        # rotation + scale matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        # translate so eyes_center maps to desired center in output
        tx = self.target_w * 0.5 - eyes_center[0]
        ty = self.target_h * self.vertical - eyes_center[1]
        # incorporate translation into M
        M[0,2] += tx
        M[1,2] += ty

        border_mode = cv2.BORDER_REPLICATE if self.replicate_border else cv2.BORDER_CONSTANT
        aligned = cv2.warpAffine(img, M, (self.target_w, self.target_h), borderMode=border_mode)
        return aligned

def make_video(images, out_path, width, height, fps, duration, zoom, face_y, face_x, replicate_border=False):
    aligner = FaceAligner(width, height, zoom, face_y, face_x, replicate_border=replicate_border)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    total_frames = duration * len(images)
    print(f"Writing {total_frames} frames ({len(images)} images, {duration} frames each) to {out_path}")

    for i, path in enumerate(images):
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: could not read {path}, skipping")
            continue
        
        aligned = aligner.align_image(image)
        # ensure correct channels
        if aligned.shape[2] == 4:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGRA2BGR)
        # write multiple frames for each image
        for _ in range(duration):
            writer.write(aligned)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(images)} images")
    writer.release()
    print("Done.")

# ---------- main / CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description='Facelapse tool by Evans Bonté')
    p.add_argument('folder', help='Folder containing selfies in .jpg format')
    p.add_argument('output', help='Output MP4 path')
    p.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='Output video width (default 1440)')
    p.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help='Output video height (default 1920)')
    p.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Frames per second (default 30)')
    p.add_argument('--duration', type=int, default=DEFAULT_DURATION, help='The number of frames per image (default 1)')
    p.add_argument('--zoom', type=float, default=DEFAULT_ZOOM, help='Zoom factor (default 0.8)')
    p.add_argument('--vertical', type=float, default=DEFAULT_VERTICAL, help='Vertical position of face as fraction of image height (default 0.35)')
    p.add_argument('--horizontal', type=float, default=DEFAULT_HORIZONTAL, help='Horizontal position of face as fraction of image width (default 0.35)')
    p.add_argument('--replicate-border', action='store_true', help='Use replicated edge pixels for out-of-bounds areas instead of black')
    p.add_argument('--randomize', action='store_true', help='Randomize the order of images instead of sorting by date')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder = args.folder
    out = args.output
    
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)
        
    if args.randomize:
        print(f"Randomizing images in {folder}")
    else:
        print(f"Sorting images in {folder}")
    images = sort_images(folder, randomize=args.randomize)
    if len(images) == 0:
        print("No images found in folder.")
        sys.exit(1)
    
    make_video(images, out, width=args.width, height=args.height, fps=args.fps, duration=args.duration, zoom=args.zoom, face_y=args.vertical, face_x=args.horizontal, replicate_border=args.replicate_border)
