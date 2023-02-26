#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import tarfile
import urllib

import cv2
import gradio as gr
import huggingface_hub
import numpy as np

TITLE = 'nagadomi/lbpcascade_animeface'
DESCRIPTION = 'This is an unofficial demo for https://github.com/nagadomi/lbpcascade_animeface.'

HF_TOKEN = os.getenv('HF_TOKEN')


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=HF_TOKEN)
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
    image_path = cv2.imread(image_path)
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    preds = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(24, 24))

    res = image_path.copy()
    for x, y, w, h in preds:
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return res[:, :, ::-1]


image_paths = load_sample_image_paths()
examples = [[path.as_posix()] for path in image_paths]

detector = load_model()
func = functools.partial(detect, detector=detector)

gr.Interface(
    fn=func,
    inputs=gr.Image(label='Input', type='filepath'),
    outputs=gr.Image(label='Output', type='numpy'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch(show_api=False)
