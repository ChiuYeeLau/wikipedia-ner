"""
Run the stanford classifier with a specified model on the test corpus.

Calculate metrics of performance.

cut -f 1,3 test.conll > test2.conll
"""
import csv
import argparse
import shlex
import subprocess
import os
import utils

from contextlib import nested
from sklearn.metrics import classification_report


def parse_arguments():
    """Reads arguments from stdin using the argsparse library."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_path', '-c', type=unicode,
                        help='Name of file with the classifier to apply.')
    parser.add_argument('--test_filepath', '-t', type=unicode,
                        help='Name of file with the test dataset.')
    parser.add_argument('--output_dirpath', '-o', type=unicode,
                        help='Directory to save the tagged documents.')
    parser.add_argument('--split_size', type=int, default=10**6,
                        help='Maximum size of the test file. If exceeded'
                             'it will be splitted.')
    parser.add_argument('--stanford_libs_path', type=unicode,
                        help='Split the test corpus into parts.')
    parser.add_argument('--task', type=unicode, default='ner',
                        help='Name of the task to evaluate.')

    return parser.parse_args()


class StanfordEvaluator(object):
    """Applies a classifier in a test corpus and evaluates the performace."""
    def __init__(self, input_filepath, stanford_libs_path,
                 classifier_path, task):
        self.input_filepath = input_filepath
        self.splitted_input_filepaths = []
        self.stanford_libs_path = stanford_libs_path
        java_lib_names = ['stanford-ner.jar', 'lib/*']
        self.java_libs = ':'.join([self.stanford_libs_path + lib_name
                                   for lib_name in java_lib_names])
        self.classifier_path = classifier_path
        self.task = task
        self._target_indices = self._build_target_indices(task)

    @staticmethod
    def _build_target_indices(task):
        """Map the possible class names to indices according to self.task

        Applicable to ner and person tasks only."""
        if task == 'ner':
            return {'I': 0, 'O': 1}
        elif task == 'person':
            return {'not_person': 0, 'O': 1, 'person': 2}
        return {}

    def get_predictions(self, input_filepath, output_filepath):
        """Run the classifier to evaluate the file in input_filepath.

        Waits until the process has finished. Saves the predictions into
        output_filepath."""
        evaluation_command = """java -cp
            {} edu.stanford.nlp.ie.crf.CRFClassifier
            -loadClassifier {}
            -plainTextDocumentReaderAndWriter
                edu.stanford.nlp.sequences.CoNLLDocumentReaderAndWriter
            -testFile {}"""

        output_file = open(output_filepath, 'w')
        process = subprocess.Popen(shlex.split(evaluation_command.format(
            self.java_libs, self.classifier_path, input_filepath,
            output_filepath)), stdout=output_file)


        # Finish all process and show the promp again. Optional.
        process.wait()
        if process.returncode:
            print "Error: Classifier application failed."
            return False
        return True

    def _write_new_file(self, split_number, input_file, split_max_size):
        """Writes the next split into a new output file."""
        output_filepath = self.input_filepath + '.part{}'.format(split_number)
        lines_read = 0
        line = None
        with open(output_filepath, 'w') as output_file:
            self.splitted_input_filepaths.append(output_filepath)
            while not (line == '\n' and lines_read > split_max_size):
                line = input_file.readline()
                if not line:  # EOF for input file.
                    break
                lines_read += 1
                output_file.write(line)
        print 'Creating split {} with {} lines'.format(split_number, lines_read)
        return lines_read

    def split(self, split_max_size):
        """Split the test file if the number of lines exceeds split_max_size.
        """
        # Count lines to see if split is necessary
        test_size = utils.count_lines_in_file(self.input_filepath)
        if test_size <= split_max_size:
            self.splitted_input_filepaths = [self.input_filepath]
            return

        with open(self.input_filepath, 'r') as input_file:
            lines_read = 0
            for split_number in range(test_size / split_max_size + 1):
                lines_read += self._write_new_file(
                    split_number, input_file, split_max_size)
                if lines_read >= test_size:
                    break

    def transform_prediction(self, prediction):
        """Adds the prediction to the confusion matrix according to self.task

        Predictions is the list with the parsed prediction. The first element
        is the named entitiy, the second the true tag, the third the
        person/organization/etc. tag.
        """
        if prediction[1] == 'O':
            return 'O'
        if self.task == 'ner':
            return 'I' if prediction[2] != 'O' else prediction[2]
        elif self.task == 'person':
            if prediction[2] == 'PERSON':
                return 'person'
            elif prediction[2] != 'O':
                return 'not_person'
        elif self.task == 'categories':
            return prediction[2]
        return 'O'

    def read_predictions(self, output_filepath, y_true, y_predicted):
        """Reads and transform predictions according to self.task."""
        with open(output_filepath, 'r') as output_file:
            prediction_reader = csv.reader(output_file, delimiter='\t')
            for prediction in prediction_reader:
                if len(prediction) == 0:
                    continue
                y_true.append(self._target_indices[prediction[1]])
                y_predicted.append(self._target_indices[
                    self.transform_prediction(prediction)])

    def get_metric(self, input_filepath, output_dirpath):
        """Compares the ground truth in files to the output prediction."""
        y_true = []
        y_predicted = []
        for input_filepath in self.splitted_input_filepaths:
            output_filepath = os.path.join(output_dirpath,
                                           os.path.basename(input_filepath))
            self.read_predictions(output_filepath, y_true, y_predicted)
        print classification_report(
            y_true, y_predicted,
            target_names=sorted(self._target_indices.keys()))

    def evaluate(self, output_dirpath, split_max_size):
        """
        Applies the classifier
        Reads the input and the output
        Calculates the metric
        """
        self.split(split_max_size)
        for input_filepath in self.splitted_input_filepaths:
            output_filepath = os.path.join(output_dirpath,
                                           os.path.basename(input_filepath))
            if not self.get_predictions(input_filepath, output_filepath):
                return

        self.get_metric(self.input_filepath, output_dirpath)


def main():
    """Main function of script."""

    args = parse_arguments()

    evaluator = StanfordEvaluator(args.test_filepath, args.stanford_libs_path,
                                  args.classifier_path, args.task)
    evaluator.evaluate(args.output_dirpath, args.split_size)


if __name__ == '__main__':
    main()
