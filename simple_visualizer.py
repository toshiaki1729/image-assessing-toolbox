import os
import csv
from typing import IO
import argparse
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt


def load_fd(f:IO):
    reader = csv.reader(f, delimiter=',')
    names = []
    values = []
    classifiers = []
    for i, row in enumerate(reader):
        if i == 0:
            names = row[1:]
        else:
            classifiers.append(row[0])
            values.append([float(v) for v in row[1:]])
    return names, classifiers, values


def visualize_fd(data, title):
    names, classifiers, values = data
    margin = 0.2
    total_w = 1 - margin
    n_cls = len(classifiers)
    w = total_w / n_cls
    gs = []
    xs = np.arange(len(names))
    fig = plt.figure(figsize=[n_cls*2+2, 8])
    for i, cf in enumerate(classifiers):
        xp = w*(i-n_cls/2 + 1)
        g = plt.bar(x=xs+xp, height=values[i], width=w, align='center', label=cf)
        for j, x in enumerate(xs):
            if j < len(values[i]):
                v = values[i][j]
                plt.text(x+xp, v, f'{v:.1f}', ha='center')
        gs.append(g)

    plt.xticks(xs+w/2, names)
    plt.legend(handles=gs, loc='best', shadow=True)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    return fig


def load_pr(f:IO):
    reader = csv.reader(f, delimiter=',')
    names = []
    xs = []
    ys = []
    classifiers = []
    for i, row in enumerate(reader):
        if i == 0:
            names = row[2:]
        else:
            if row[1] == 'precision':
                classifiers.append(row[0])
                ys.append([float(v) for v in row[2:]])
            elif row[1] == 'recall':
                xs.append([float(v) for v in row[2:]])
    return names, classifiers, xs, ys


def visualize_pr(data, title):
    names, classifiers, xs, ys = data
    MARKERS = ['o', '^', 's', '*', 'v', 'X', 'P', 'D', '<', '>', 'd']
    fig = plt.figure(figsize=[8, 8])
    
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()['color']
    for i, cf in enumerate(classifiers):
        for j, name in enumerate(names):
            if j < len(xs[i]) and len(ys[i]):
                plt.scatter(xs[i][j], ys[i][j], c=COLORS[j%len(COLORS)], marker=MARKERS[(i+j//len(COLORS))%len(MARKERS)], label=f'{cf} - {name}')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best', shadow=True)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    return fig


Visualizer = namedtuple('Visualizer', ['dataloader', 'visualizer'])
VISUALIZERS = {
    'frechet_distance' : Visualizer(load_fd, visualize_fd),
    'precision_recall' : Visualizer(load_pr, visualize_pr)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_type', type=str, choices=['frechet_distance', 'precision_recall'], help='Result csv type')
    parser.add_argument('result_path', type=str, help='Result csv path to load')
    parser.add_argument('--title', type=str, default=None, help='Graph title')
    parser.add_argument('--out_path', type=str, default=None, help='Path to output image file')
    parser.add_argument('--out_format', type=str, default='png', help='Output image format')
    parser.add_argument('--out_dpi', type=int, default=150, help='Output image resolution (dpi)')
    return parser.parse_args()


def main(opts: argparse.Namespace):
    with open(os.path.abspath(opts.result_path), mode='r', encoding='utf8', newline='') as f:
        data = VISUALIZERS[opts.result_type].dataloader(f)
    
    g = VISUALIZERS[opts.result_type].visualizer(data, opts.title)
    if opts.out_path:
        g.savefig(os.path.abspath(opts.out_path), format=opts.out_format, dpi=opts.out_dpi)

if __name__ == '__main__':
    main(parse_args())