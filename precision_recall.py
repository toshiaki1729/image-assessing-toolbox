import os
import argparse
from typing import Dict, List
import csv
import json

import numpy as np
from tqdm import tqdm

from improved_precision_recall import ManifoldEstimator, get_precision_and_recall, Manifold, PrecisionRecall
from utilities import extract_from_filename, glob_filepaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_dir', type=str, help='Real features directory (load all npz files inside)')
    parser.add_argument('fake_dir', type=str, help='Fake features directory (load all npz files inside)')
    parser.add_argument('out_path', type=str, help='Path to output csv file')
    parser.add_argument('--recursive', '-r', action='store_true', help='Load features recursively from given directory')
    parser.add_argument('-k', type=int, default=3, help='K-value used for estimating the manifold to which the dataset belongs in the feature space (feature vectors closer than k-neighbors of any dataset are approximated as belonging to the manifold)')
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Which device to use')
    parser.add_argument('--json', '-j', type=str, nargs='?', default=None, const='', help='Output json file')
    return parser.parse_args()


def main(opts):
    real_dir = os.path.abspath(opts.real_dir)
    real_files = glob_filepaths(real_dir, recursive=opts.recursive)
    real_manifold_dict = dict()
    classifier_set = set()

    estimator = ManifoldEstimator(k=opts.k, device=opts.device)

    print('Loading real features and estimating manifolds...')
    for f in tqdm(real_files):
        name, classifier = extract_from_filename(f)
        features = np.load(f)['features']
        real_manifold_dict[classifier] = estimator.evaluate(features)
        classifier_set.add(classifier)

    fake_dir = os.path.abspath(opts.fake_dir)
    fake_files = glob_filepaths(fake_dir, recursive=opts.recursive)
    fake_manifold_list_dict:Dict[List[Manifold]] = dict()

    fake_name_set = set()

    print('Loading fake features and estimating manifolds...')
    for f in tqdm(fake_files):
        name, classifier = extract_from_filename(f)
        features = np.load(f)['features']
        manifold = estimator.evaluate(features)
        if classifier in fake_manifold_list_dict:
            fake_manifold_list_dict[classifier] += [(name, manifold)]
        else:
            fake_manifold_list_dict[classifier] = [(name, manifold)]
        fake_name_set.add(name)

    print('Evaluating Precision and Recall...')
    results = dict()
    classifiers = tqdm(real_manifold_dict.keys())
    classifiers.set_description('Total progress')
    for classifier in classifiers:
        real_manifold = real_manifold_dict[classifier]
        if classifier not in results:
            results[classifier] = dict()
        fake_manifold_list = tqdm(fake_manifold_list_dict.get(classifier, []))
        for name, fake_manifold in fake_manifold_list:
            fake_manifold_list.set_description(f'Classifier={classifier}, Name={name}')
            results[classifier][name] = get_precision_and_recall(real_manifold, fake_manifold)
    
    save_path = os.path.abspath(opts.out_path)
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'Writing results to {save_path}')
    with open(save_path, mode='w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        fake_names = sorted(fake_name_set)
        writer.writerow(['']*2 + [name for name in fake_names])
        for classifier in sorted(classifier_set):
            rows = [[classifier, 'precision'], [classifier, 'recall']]
            res_cls = results.get(classifier)
            if res_cls:
                for name in fake_names:
                    v:PrecisionRecall = res_cls.get(name)
                    rows[0].append(v.precision)
                    rows[1].append(v.recall)
                    
            writer.writerows(rows)
    
    if opts.json is not None:
        json_path = os.path.splitext(save_path)[0] + '.json' if not opts.json else opts.json
        print(f'Writing results to {json_path}')
        json_data = []
        for classifier in sorted(classifier_set):
            res_cls = results.get(classifier)
            if not res_cls:
                continue
            for name in fake_names:
                value = res_cls.get(name)
                if value is None:
                    continue
                json_data.append({'classifier':classifier, 'name':name, 'value': value})
        with open(json_path, mode='w', encoding='utf8', newline='') as f:
            json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    main(parse_args())