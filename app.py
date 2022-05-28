#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import tarfile
import urllib

import cv2
import gradio as gr
import huggingface_hub
import numpy as np

ORIGINAL_REPO_URL = 'https://github.com/nagadomi/lbpcascade_animeface'
TITLE = 'nagadomi/lbpcascade_animeface'
DESCRIPTION = f'A demo for {ORIGINAL_REPO_URL}'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model() -> cv2.CascadeClassifier:
    url = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
    path = pathlib.Path('lbpcascade_animeface.xml')
    if not path.exists():
        urllib.request.urlretrieve(url, path.as_posix())
    return cv2.CascadeClassifier(path.as_posix())


def detect(image, detector: cv2.CascadeClassifier) -> np.ndarray:
    image = cv2.imread(image.name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preds = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(24, 24))

    res = image.copy()
    for x, y, w, h in preds:
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return res[:, :, ::-1]


def main():
    gr.close_all()

    args = parse_args()

    image_paths = load_sample_image_paths()
    examples = [[path.as_posix()] for path in image_paths]

    detector = load_model()
    func = functools.partial(detect, detector=detector)
    func = functools.update_wrapper(func, detect)

    gr.Interface(
        func,
        gr.inputs.Image(type='file', label='Input'),
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
