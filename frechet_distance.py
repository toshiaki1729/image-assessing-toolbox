import os
import argparse
from typing import Dict, List
import csv

from tqdm import tqdm
import numpy as np

from pytorch_fid.fid_score import calculate_frechet_distance

from features import Statistics
from utilities import extract_from_filename, glob_filepaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('real_dir', type=str, help='Real features directory (load all npz files inside)')
    parser.add_argument('fake_dir', type=str, help='Fake features directory (load all npz files inside)')
    parser.add_argument('out_path', type=str, help='Path to output csv file')
    parser.add_argument('--recursive', '-r', action='store_true', help='Load features recursively from given directory')
    return parser.parse_args()


def main(opts):
    real_dir = os.path.abspath(opts.real_dir)
    real_files = glob_filepaths(real_dir, recursive=opts.recursive)
    real_stats_dict = dict()
    
    classifier_set = set()

    print('Loading real features...')
    for f in tqdm(real_files):
        name, classifier = extract_from_filename(f)
        features = np.load(f)['features']
        real_stats_dict[classifier] = Statistics(name, classifier, features)
        classifier_set.add(classifier)

    fake_dir = os.path.abspath(opts.fake_dir)
    fake_files = glob_filepaths(fake_dir, recursive=opts.recursive)
    fake_stats_list_dict:Dict[List[Statistics]] = dict()

    fake_name_set = set()

    print('Loading fake features...')
    for f in tqdm(fake_files):
        name, classifier = extract_from_filename(f)
        features = np.load(f)['features']
        stats = Statistics(name, classifier, features)
        if classifier in fake_stats_list_dict:
            fake_stats_list_dict[classifier] += [stats]
        else:
            fake_stats_list_dict[classifier] = [stats]
        fake_name_set.add(name)

    print('Evaluating Frechet Distances...')
    results = dict()
    classifiers = tqdm(real_stats_dict.keys())
    classifiers.set_description('Total progress')
    for classifier in classifiers:
        real_stats = real_stats_dict[classifier]
        if classifier not in results:
            results[classifier] = dict()
        fake_stats_list = tqdm(fake_stats_list_dict.get(classifier, []))
        for fake_stats in fake_stats_list:
            fake_stats_list.set_description(f'Classifier={classifier}, Name={fake_stats.name}')
            results[classifier][fake_stats.name] = calculate_frechet_distance(real_stats.mu, real_stats.sigma, fake_stats.mu, fake_stats.sigma)
    
    save_path = os.path.abspath(opts.out_path)
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'Writing results to {save_path}')
    with open(save_path, mode='w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        fake_names = sorted(fake_name_set)
        writer.writerow([''] + [name for name in fake_names])
        for classifier in sorted(classifier_set):
            row = [classifier]
            res_cls = results.get(classifier)
            if res_cls:
                for name in fake_names:
                    row.append(res_cls.get(name))
            writer.writerow(row)


if __name__ == '__main__':
    main(parse_args())