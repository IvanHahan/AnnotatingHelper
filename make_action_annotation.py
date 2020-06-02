import numpy
import os
import json
import argparse
from utils import parse_annotation, make_dir_if_needed
import cv2
import re
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/UnicornKing/20180101_120040')
parser.add_argument('--output_image_dir', default='data/images')
parser.add_argument('--output_label_path', default='data/label.txt')
args = parser.parse_args()

if __name__ == '__main__':
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    label_counts = {}
    records = []
    for file, annot in tqdm(annotations):
        file_name = os.path.splitext(file)[0]
        for i, shape in enumerate(annot['shapes']):
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            frame = cv2.imread(os.path.join(args.data_dir, file_name + '.jpg'))[y1:y2, x1:x2]
            frame_path = os.path.join(args.output_image_dir, file_name + f'_{i}.jpg')
            cv2.imwrite(frame_path, frame)

            frame_index = label_counts.get(shape['label'], -1) + 1
            label_counts[shape['label']] = frame_index

            label = shape['label']

            records.append((frame_path, frame_index, label))

    records = np.array(records)
    keep_labels, counts = np.unique(records[:, 2], return_counts=True)
    keep_labels = keep_labels[counts > 1]
    # print(np.apply_along_axis(lambda x: x in keep_labels, 0, records[:, 2]).shape)
    records = records[[True if label in keep_labels else False for label in records[:, 2]]]
    records = np.array([(path, idx, re.sub(r'_\d+', '', label)) for (path, idx, label) in records])

    unique_labels = np.unique(records[:, 2])
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    records = [(path, idx, label_to_idx[label]) for (path, idx, label) in records]
    records = [f'{path}, {idx}, {label}' for (path, idx, label) in records]

    with open(args.output_label_path, 'w') as f:
        f.write('\n'.join(records))
