"""
Calculate the value of the ttest for files with paired predictions.

The argument is a file with the name of the pairs of predictions files. Each
prediction file has a prediction per line. The prediction class has to be an
integer.
"""

import argparse
import numpy as np

from scipy import stats


def read_filename_pairs(paired_filenames):
    """Reads tuples of filenames from file with name paired_filenames."""
    filenames = []
    with open(paired_filenames, 'r') as file_:
        for line in file_:
            filenames.append(line.split())
    return filenames


def read_prediction(filename):
    """Returns an array-like object with the predictions from the file filename.
    """
    with open(filename, 'r') as file_:
        return [int(prediction) for prediction in file_.read().split()]


def significance_values(filename1, filename2, true_values):
    """Shows the paired t_test between the predictions in the filenames."""
    prediction1 = (np.array(read_prediction(filename1)) == true_values).astype(np.int32)
    prediction2 = (np.array(read_prediction(filename2)) == true_values).astype(np.int32)

    if len(prediction1) != len(prediction2):
        print 'Error: predictions have different sizes.'
        return

    return stats.ttest_rel(prediction1, prediction2)


def main(paired_filenames, true_values):
    filename_pairs = read_filename_pairs(paired_filenames)

    true_values = np.loadtxt(true_values)

    for filename1, filename2 in filename_pairs:
        t_statistic, p_value = significance_values(filename1, filename2, true_values)
        print filename1, 'vs', filename2
        print 'T Statistic', t_statistic
        print 'P Value', p_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paired_filenames', type=unicode)
    parser.add_argument('true_values', type=unicode)

    args = parser.parse_args()

    main(args.paired_filenames, args.true_values)
