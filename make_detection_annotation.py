import numpy
import os
import json
import argparse
from utils import parse_annotation, make_dir_if_needed
import cv2
import re
import numpy as np
from tqdm import tqdm
from definitions import object_to_idx, idx_to_object

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/UnicornKing/20180101_120040')
parser.add_argument('--output_image_dir', default='data/detection_images')
parser.add_argument('--output_label_path', default='data/detection_label.txt')
args = parser.parse_args()

if __name__ == '__main__':
    make_dir_if_needed(args.output_image_dir)
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    label_counts = {}
    records = []
    dirs = set()
    with open(args.output_label_path, 'w') as f:
        for file, annot in tqdm(annotations):
            file_name = os.path.splitext(file)[0]
            image_name = file_name + '.jpg'
            for i, shape in enumerate(annot['shapes']):
                label = shape['label']
                if label not in object_to_idx:
                    continue
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]

                if x1 == x2 or y1 == y2:
                    continue

                x1 = min(x1, x2)
                x2 = max(x1, x2)
                y1 = min(y1, y2)
                y2 = max(y1, y2)

                frame = cv2.imread(os.path.join(args.data_dir, image_name))
                frame_path = os.path.join(args.output_image_dir, image_name)

                cv2.imwrite(frame_path, frame)
                f.write(f'{image_name} {x1} {y1} {x2} {y2} {object_to_idx[label]}\n')

