"""Script to generate a weighted random prediction for a test dataset."""
import argparse

import logging
import numpy
import pandas
import os

from sklearn import metrics

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_filepath', '-f,', type=str,
                        help='Path of csv file with the test labels.')
    parser.add_argument('--results_dirname', '-d,', type=str,
                        help='Path of directory to store the results file.')
    return parser.parse_args()

def main():
    args = read_arguments()
    test_dataset = pandas.read_csv(args.labels_filepath)
    if test_dataset.columns.tolist() != ['true', 'prediction']:
        logging.error('Labels file must be a csv file with two numeric columns:'
                      'true and prediction.')
        return
    elements, counts = numpy.unique(test_dataset.true, return_counts=True)
    counts = (counts / test_dataset.true.shape[0]).astype(numpy.float64)
    random_predictions = numpy.random.choice(
        elements, size=test_dataset.true.shape, p=counts)

    test_dataset.prediction = random_predictions
    test_dataset.to_csv(os.path.join(args.results_dirname,
                                     'evaluation_predictions_random.csv'),
                        index=False)

    test_results = []
    test_results.append(
        ['accuracy', metrics.accuracy_score(test_dataset.true,
                                            test_dataset.prediction)])
    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(
        test_dataset.true, test_dataset.prediction, average='macro')
    test_results.append(['prec-macro', prec])
    test_results.append(['recall-macro', recall])
    test_results.append(['fscore-macro', fscore])

    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(
        test_dataset.true, test_dataset.prediction, average='micro')
    test_results.append(['prec-micro', prec])
    test_results.append(['recall-micro', recall])
    test_results.append(['fscore-micro', fscore])

    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(
        test_dataset.true, test_dataset.prediction, average='weighted')
    test_results.append(['prec-weighted', prec])
    test_results.append(['recall-weighted', recall])
    test_results.append(['fscore-weighted', fscore])

    test_results = pandas.DataFrame(test_results,
                                    columns=['metrics', 'values'])
    test_results.to_csv(os.path.join(args.results_dirname,
                                     'evaluation_results_random.csv'),
                        index=False)



if __name__ == '__main__':
    main()
