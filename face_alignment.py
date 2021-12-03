import numpy as np
import scipy.ndimage
import PIL.Image
import PIL.ImageFile
import multiprocessing
from functools import partial
import os
import argparse
from utils import load_json
from tqdm import tqdm

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error


def image_align(img, landmarks, output_size=1024, transform_size=4096, enable_padding=True):
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    eye_left = np.array(landmarks['eye_left'])
    eye_right = np.array(landmarks['eye_right'])
    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])

    # Calculate auxiliary vectors.
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


def align_worker(files, raw_dir, save_dir):
    aligned_name, raw_name, landmarks = files
    raw_img_path = os.path.join(raw_dir, raw_name)
    aligned_img_path = os.path.join(save_dir, aligned_name)

    if not os.path.exists(raw_img_path):
        print('original image not exist: %s' % raw_img_path)
        return 0
    if os.path.exists(aligned_img_path):
        return 1

    img = PIL.Image.open(raw_img_path)
    img_aligned = image_align(img, landmarks, output_size=1024, transform_size=4096, enable_padding=True)
    img_aligned.save(aligned_img_path)
    return 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_dir', type=str, default='./AAHQ-dataset.json', help='path to AAHQ metadata file')
    parser.add_argument('--raw_dir', type=str, default='./raw', help='path to original images')
    parser.add_argument('--save_dir', type=str, default='./aligned', help='path to save aligned images')
    parser.add_argument('--n_worker', type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    js = load_json(args.json_dir)

    imgnames = js.keys()
    files = [(aligned_name, js[aligned_name]['raw_name'], js[aligned_name]['landmarks']) for aligned_name in imgnames]

    align_fn = partial(align_worker, raw_dir=args.raw_dir, save_dir=args.save_dir)

    total = 0
    with multiprocessing.Pool(args.n_worker) as pool:
        for code in tqdm(pool.imap_unordered(align_fn, files)):
            if code > 0:
                total += 1

    print('num of saved images: ', total)
    print('Done!')
