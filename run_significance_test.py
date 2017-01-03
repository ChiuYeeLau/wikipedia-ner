"""
Calculate the value of the ttest for files with paired predictions.

The argument is a file with the name of the pairs of predictions files. Each
prediction file has a prediction per line. The prediction class has to be an
integer.
"""

from __future__ import print_function, unicode_literals
import argparse
import pandas

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
    return pandas.read_csv(filename).prediction


def significance_values(filename1, filename2):
    """Shows the paired t_test between the predictions in the filenames."""
    prediction1 = read_prediction(filename1)
    prediction2 = read_prediction(filename2)

    if len(prediction1) != len(prediction2):
        print('Error: predictions have different sizes.')
        return

    return stats.ttest_rel(prediction1, prediction2)


def main(args):
    if args.paired_filenames is None:
        assert args.filename1 is not None and args.filename2 is not None
        filename_pairs = [(args.filename1, args.filename2)]
    else:
        filename_pairs = read_filename_pairs(args.paired_filenames)
    for filename1, filename2 in filename_pairs:
        t_statistic, p_value = significance_values(filename1, filename2)
        print('{} vs {}'.format(filename1, filename2))
        print('T Statistic {}'.format(t_statistic))
        print('P Value {}'.format(p_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired_filenames', '-f', type=str, default=None,
                        help='A file with paired names of path to csv '
                             'prediction files. Each pair will be compared'
                             'against the other.')
    parser.add_argument('--filename1', '-1', type=str, default=None,
                        help='If no paired filename is given, this is the name'
                             'of the file with csv predictions to compare'
                             'against filename2.')
    parser.add_argument('--filename2', '-2', type=str, default=None,
                        help='If no paired filename is given, this is the name'
                             'of the file with csv predictions to compare'
                             'against filename2.'
                        )

    args = parser.parse_args()

    main(args)
