# Code for computing IS / FID stats

import argparse
import os
import numpy as np


def parse_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    is_score_mean = 0
    is_score_std = 0
    fid_score = 0
    intra_fid_score = 0

    for line in lines:
        if line.startswith('Best IS:'):
            lsplit = line.split(' ')
            is_score_mean = float(lsplit[2])
            is_score_std = float(lsplit[4])
        elif line.startswith('Best FID:'):
            lsplit = line.split(' ')
            fid_score = float(lsplit[2])
        elif line.startswith('Best intra FID:'):
            lsplit = line.split(' ')
            intra_fid_score = float(lsplit[3])
    return is_score_mean, is_score_std, fid_score, intra_fid_score


def main():
    parser = argparse.ArgumentParser(description='Arg parser')
    parser.add_argument('--results_dir', default='', type=str,
                        help='path to config file')
    args = parser.parse_args()

    fols = {}
    dirs = os.listdir(args.results_dir)

    for d in dirs:
        flist = d.split('_')[1:-3]
        for i, fpart in enumerate(flist):
            if i == 0:
                fname = fpart
            else:
                fname = fname + '_' + fpart

        logdir = os.path.join(args.results_dir, d, 'log.txt')
        is_mean, is_std, fid, intra_fid = parse_file(logdir)

        if is_mean > 0 and is_std > 0 and fid > 0:
            if fname not in fols:
                fols[fname] = [(is_mean, is_std, fid, intra_fid)]
            else:
                fols[fname].append((is_mean, is_std, fid, intra_fid))
                
    print('\n\n')
    for fname in fols:
        is_mean = np.mean(np.array([score[0] for score in fols[fname]]))
        is_std = np.std(np.array([score[1] for score in fols[fname]]))
        fid = np.mean(np.array([score[2] for score in fols[fname]]))
        intra_fid = np.mean(np.array([score[3] for score in fols[fname]]))

        print(fname)
        print('IS: {} +/- {}'.format(is_mean, is_std))
        print('FID: {}'.format(fid))
        if intra_fid > 0:
            print('Intra FID: {}'.format(intra_fid))
        print('#####################\n\n')


if __name__ == '__main__':
    main()



