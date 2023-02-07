import os
from glob import glob


def extract_from_filename(path):
    filename, _ = os.path.splitext(os.path.basename(path))
    names = filename.split('_')
    name = '_'.join(names[:-1])
    classifier = names[-1]
    return name, classifier


def glob_filepaths(path, fileext='.npz', recursive=False):
    return [p for p in glob(os.path.join(path, f'**/*{fileext}' if recursive else f'*{fileext}'), recursive=recursive) if os.path.isfile(p)]