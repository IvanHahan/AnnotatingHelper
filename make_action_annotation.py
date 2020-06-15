import numpy
import os
import json
import argparse
from utils import parse_annotation, make_dir_if_needed
import cv2
import re
import numpy as np
from tqdm import tqdm
from definitions import action_to_idx, idx_to_action

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/UnicornKing/20180101_120040')
parser.add_argument('--output_image_dir', default='data/actions')
parser.add_argument('--output_label_path', default='data/actions.txt')
args = parser.parse_args()

if __name__ == '__main__':
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    label_counts = {}
    records = []
    dirs = set()
    for file, annot in tqdm(annotations):
        file_name = os.path.splitext(file)[0]
        for i, shape in enumerate(annot['shapes']):
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            label = shape['label']
            if re.sub(r'_\d+', '', label) not in action_to_idx:
                continue

            frame = cv2.imread(os.path.join(args.data_dir, file_name + '.jpg'))[y1:y2, x1:x2]
            frame_dir = os.path.join(args.output_image_dir, label)
            make_dir_if_needed(frame_dir)

            frame_index = label_counts.get(shape['label'], 0) + 1
            label_counts[shape['label']] = frame_index

            frame_path = os.path.join(frame_dir, '{:08d}.jpg'.format(frame_index))
            cv2.imwrite(frame_path, frame)

            dirs.add(label)

    with open(args.output_label_path, 'w') as f:
        for dir in dirs:
            frame_number = len([x for x in os.listdir(os.path.join(args.output_image_dir, dir))
                            if os.path.splitext(x)[1] == '.jpg'])
            label = re.sub(r'_\d+', '', dir)
            f.write(f'{dir} {frame_number} {action_to_idx[label]}\n')
