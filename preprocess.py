import argparse
import glob
import os
import traceback
import inspect

import numpy as np

import image_classifiers as imc


def derived_class_list(module, base):
    return [name for name, _ in inspect.getmembers(module, lambda x: inspect.isclass(x) and issubclass(x, base) and x is not base)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('out_dir', type=str, help='Directory to save feature vectors as npz file')
    parser.add_argument('classifiers', type=str, nargs='+', choices=derived_class_list(imc, imc.Classifier), help='Name of classifiers to use')
    
    parser.add_argument('--out_filename', '-o', type=str, default='result', help='Output base filename (save result with a name like "[out filename]_[classifier name].npz")')
    parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=50, help='How many samples per batch to load (for each classifiers, 1 or more values)')
    parser.add_argument('--num_workers', '-n', type=int, nargs='+', default=8, help='How many subprocesses to use for data loading (for each classifiers, 1 or more values)')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='Which device to use (cuda, cpu, ...)')
    parser.add_argument('--image_resize_repeat', action='store_true', help='Repeat edge color on reshaping images to prevent from changing aspect ratio')
    
    def parse_multi_args(opts, argname, default, min_length):
        arg = getattr(opts, argname)
        if isinstance(arg, list):
            if len(arg) == 1:
                print(f'{argname}={arg[0]} will be used for all classifiers.')
                arg = arg * min_length
            elif len(arg) < min_length:
                print(f'The number of batch sizes are fewer than the classifiers. Default value {default} will be used with remainings.')
                arg = arg + [default]*(min_length - len(arg))
        elif isinstance(arg, int):
            print(f'{argname}={arg} will be used for all classifeirs.')
            arg = [arg] * min_length
        return arg
    
    opts = parser.parse_args()
    opts.batch_size = parse_multi_args(opts, 'batch_size', 50, len(opts.classifiers))
    opts.num_workers = parse_multi_args(opts, 'num_workers', 8, len(opts.classifiers))
    return opts


def main(opts:argparse.Namespace):
    files = [p for p in glob.glob(os.path.join(os.path.abspath(opts.data_dir), '*')) if os.path.isfile(p)]

    for i, cname in enumerate(opts.classifiers):
        if not hasattr(imc, cname):
            print(f'No such classifer is implemented: {cname}')
            continue
        classifier: imc.Classifier = getattr(imc, cname)
        
        with classifier(device=opts.device) as c:
            try:
                result = c(files, opts.batch_size[i], opts.num_workers[i])
                savedir = os.path.abspath(opts.out_dir)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                np.savez_compressed(os.path.join(savedir, opts.out_filename + '_' + cname + '.npz'), features=result)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                continue

if __name__ == '__main__':
    main(parse_args())