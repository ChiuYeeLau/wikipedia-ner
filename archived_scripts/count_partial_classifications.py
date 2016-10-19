"""Script to count the number of partially recognized entities in a file.

The file should have format:
word \t true_label \t predicted_label
"""

import argparse
import csv
import os


def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dirname', type=unicode,
                        help='Path of file to preprocess')

    return parser.parse_args()


def main():
    """Main function of the script."""
    args = read_arguments()

    filenames = sorted(filter(
        lambda f: os.path.isfile(os.path.join(args.input_dirname, f)),
        os.listdir(args.input_dirname)))
    file_paths = [os.path.join(args.input_dirname, filename)
                  for filename in filenames]

    # Counters
    counters = {'person': 0, 'not_person': 0, 'O': 0}
    recognized_counter = {'person': 0, 'not_person': 0, 'O': 0}
    partial_counter = {'person': 0, 'not_person': 0, 'O': 0}
    missed_counter = {'person': 0, 'not_person': 0, 'O': 0}

    for filename in file_paths:
        recognized = True
        partial = False
        prev_line = []
        with open(filename, 'r') as input_file:
            reader = csv.reader(input_file, delimiter='\t')
            for line in reader:
                # End of document or first 'O' after an entity
                if len(line) == 0 or (line[1] == 'O' and len(prev_line)
                                      and prev_line[1] != 'O'):
                    counters[prev_line[1]] += 1
                    if recognized:
                        recognized_counter[prev_line[1]] += 1
                    elif partial:
                        partial_counter[prev_line[1]] += 1
                    else:  # Not even partially recognized
                        missed_counter[prev_line[1]] += 1
                    recognized = True
                    partial = False
                    prev_line = line
                    continue
                if line[1] == 'O':
                    prev_line = line
                    continue
                # Inside an entity
                if line[1] == line[2]:
                    partial = True  # At least it recognized something.
                if line[1] != line[2]:
                    recognized = False  # Not all entity is recognized
                prev_line = line

    print 'Person'
    print 'Total entites', counters['person']
    print 'Fully Recognized', recognized_counter['person']
    print 'Partially Recognized', partial_counter['person']
    print 'Missed', missed_counter['person']
    print ''
    print 'Not Person'
    print 'Total entites', counters['not_person']
    print 'Fully Recognized', recognized_counter['not_person']
    print 'Partially Recognized', partial_counter['not_person']
    print 'Missed', missed_counter['not_person']


if __name__ == '__main__':
    main()
