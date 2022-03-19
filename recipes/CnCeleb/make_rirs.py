#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_rirs.py
@Time: 2022/3/20 02:31
@Overview:
"""
import pathlib


import pathlib
import sys


pn_dir = sys.argv[1]

pn_dir=pathlib.Path(pn_dir)

with open(str(pn_dir) + '/noise_list', 'w') as f:
    for wav in pn_dir.glob('*.wav'):
        wav_id = str(wav).split('/')[-1].split('.')[0]
        f.write('--noise-id ' + wav_id + '--noise-type point-source --bg-fg-type foreground RIRS_NOISES/pointsource_noises/' + wav_id + '.wav\n')

