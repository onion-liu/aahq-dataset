import os
import requests
import argparse
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_dir', type=str, default='./AAHQ-dataset.json', help='path to AAHQ metadata file')
    parser.add_argument('--save_dir', type=str, default='./raw', help='path to save original images')
    parser.add_argument('--retries', type=int, default=4, help='Maximum number of retries')
    parser.add_argument('--max_delay_second', type=float, default=2.0)

    args = parser.parse_args()

    json_dir = args.json_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    js = load_json(json_dir)
    urls = []
    for k, v in js.items():
        urls.append(v['image_url'])
    NUM = len(urls)
    not_found_urls = []
    for idx in range(NUM):
        url = urls[idx]
        imgname = '_'.join(url.split('/')[-5:])

        if os.path.exists(os.path.join(save_dir, imgname)):
            print('exist! [%d/%d] %s' % (idx, NUM, imgname))
            continue

        res = None
        for try_num in range(args.retries):
            randdelay(0, args.max_delay_second)

            res = requests.get(url)
            if res.status_code == 200:
                break

        if res.status_code == 200:
            img_file = res.content

            with open(os.path.join(save_dir, imgname), 'wb') as f:
                f.write(img_file)

            print('[%d/%d] %s' % (idx, NUM, imgname))
        else:
            print('Not found! [%d/%d] %s' % (idx, NUM, imgname))
            not_found_urls.append(urls)

    print('\n====================== URLs Not Found ======================')
    for url in not_found_urls:
        print(url)

    print('\nDone!')
