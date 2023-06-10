#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import tarfile
import urllib.request

import cv2
import gradio as gr
import huggingface_hub
import numpy as np

DESCRIPTION = '# [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)'


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset')
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model() -> cv2.CascadeClassifier:
    url = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
    path = pathlib.Path('lbpcascade_animeface.xml')
    if not path.exists():
        urllib.request.urlretrieve(url, path.as_posix())
    return cv2.CascadeClassifier(path.as_posix())


def detect(image_path: str, detector: cv2.CascadeClassifier) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preds = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(24, 24))

    res = image.copy()
    for x, y, w, h in preds:
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return res[:, :, ::-1]


image_paths = load_sample_image_paths()
examples = [[path.as_posix()] for path in image_paths]

detector = load_model()
fn = functools.partial(detect, detector=detector)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input', type='filepath')
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Result')

    gr.Examples(examples=examples,
                inputs=image,
                outputs=result,
                fn=fn,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')
    run_button.click(fn=fn, inputs=image, outputs=result, api_name='predict')
demo.queue(max_size=15).launch()
