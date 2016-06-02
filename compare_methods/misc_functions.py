import numpy as np

def mape_with_error(preds, truth):
    assert (not np.any(truth == 0))
    abs_diffs = np.abs((preds - truth) / (1.0 * truth))
    mape = np.mean(abs_diffs, axis=0)
    sd = np.sqrt(np.var(abs_diffs, axis=0) / abs_diffs.shape[0])
    return (mape, sd)

def mape(preds, truth):
    assert (not np.any(truth == 0))
    abs_diffs = np.abs((preds - truth) / (1.0 * truth))
    mape = np.mean(abs_diffs, axis=0)
    return mape

def n_arg_max(l, n):
  return [v[1] for v in sorted(zip(l, range(len(l))), key=lambda x: -x[0])[:n]]

def n_arg_min(l, n):
  return [v[1] for v in sorted(zip(l, range(len(l))), key=lambda x: x[0])[:n]]

def list_inds(l, inds):
    return [l[i] for i in inds]

def filters_string_to_filters(filters_string):
    if filters_string == "":
        return ({}, "")
    filter_pair_strings = filters_string.split(";")
    filter_pair_strings.sort()
    filters = {}
    for filter_pair in filter_pair_strings:
        field, min_max_string = filter_pair.split(":")
        min_max = [float(val) for val in min_max_string.split(",")]
        assert(len(min_max) == 1 or len(min_max) == 2)
        if len(min_max) == 1:
            min_max.append(float("inf"))
        filters[field] = min_max
    return (filters, "-".join(filter_pair_strings))

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

NORMAL = 0    # use python zip libraries
PROCESS = 1   # use (zcat, gzip) or (bzcat, bzip2)
PARALLEL = 2  # (pigz -dc, pigz) or (pbzip2 -dc, pbzip2)

def any_open(filename, mode='r', buff=1024*1024, external=PARALLEL):
    if 'r' in mode and 'w' in mode:
        return None
    if filename.startswith('!'):
        import subprocess
        if 'r' in mode:
            return subprocess.Popen(filename[1:], shell=True, bufsize=buff,
                                    stdout=subprocess.PIPE).stdout
        elif 'w' in mode:
            return subprocess.Popen(filename[1:], shell=True, bufsize=buff,
                                    stdin=subprocess.PIPE).stdin
    elif filename.endswith('.bz2'):
        if external == NORMAL:
            import bz2
            return bz2.BZ2File(filename, mode, buff)
        elif external == PROCESS:
            if not which('bzip2'):
                return any_open(filename, mode, buff, NORMAL)
            if 'r' in mode:
                return any_open('!bzip2 -dc ' + filename, mode, buff)
            elif 'w' in mode:
                return any_open('!bzip2 >' + filename, mode, buff)
        elif external == PARALLEL:
            if not which('pbzip2'):
                return any_open(filename, mode, buff, PROCESS)
            if 'r' in mode:
                return any_open('!pbzip2 -dc ' + filename, mode, buff)
            elif 'w' in mode:
                return any_open('!pbzip2 >' + filename, mode, buff)
    elif filename.endswith('.gz'):
        if external == NORMAL:
            import gzip
            return gzip.GzipFile(filename, mode, buff)
        elif external == PROCESS:
            if not which('gzip'):
                return any_open(filename, mode, buff, NORMAL)
            if 'r' in mode:
                return any_open('!gzip -dc ' + filename, mode, buff)
            elif 'w' in mode:
                return any_open('!gzip >' + filename, mode, buff)
        elif external == PARALLEL:
            if not which('pigz'):
                return any_open(filename, mode, buff, PROCESS)
            if 'r' in mode:
                return any_open('!pigz -dc ' + filename, mode, buff)
            elif 'w' in mode:
                return any_open('!pigz >' + filename, mode, buff)
    elif filename.endswith('.xz'):
        if which('xz'):
            if 'r' in mode:
                return any_open('!xz -dc ' + filename, mode, buff)
            elif 'w' in mode:
                return any_open('!xz >' + filename, mode, buff)
    else:
        return open(filename, mode, buff)
    return None

def dump_pickle_with_zip(obj, file_path):
    import cPickle as pickle
    with any_open(file_path, 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def read_pickle_with_zip(file_path):
    import cPickle as pickle
    with any_open(file_path, 'r') as f:
        obj = pickle.load(f)
    return obj