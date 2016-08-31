"""
This script reads a file with entities from the YAGO ontology and returns
the same entities mapped to they corresponding class in the LKIF ontology.

The input argument is the name of a directory. The script will read each
.pickle file in the directory. The name of the file must have the following
format:
YAGO_CLASSNAME-OFFSET.pickle

The result will be a dictionary mapping the name of the entity to the
LKIF corresponding class in a pickled file.
"""

import argparse
import pickle

from collections import defaultdict
from os import listdir
from os.path import isfile, join

# Mapping from YAGO categories to LKIF cathegories
# Based in the work of Laura Alonso
# Includes only the first level of the hierarchy. Each LKIF class has
YAGO_TO_LKIF_MAPPING = {
    'wordnet_legal_power_105198427': ['Potestative_Expression'],
    'wordnet_right_113341756': ['Potestative_Right'],
    'wordnet_exemption_100213903': ['Immunity'],
    'wordnet_legal_document_106479665': ['Legal_Document'],
    'wordnet_legal_code_106667792': ['Regulation', 'Code'],
    'wordnet_law_106532330': ['Regulation', 'Code'],
    'wordnet_law_108441203': ['Regulation', 'Code'],
    'wordnet_legislative_act_106564387': ['Regulation', 'Code'],
    'wordnet_contract_106520944': ['Contract'],
    'wordnet_treaty_106773434': ['Treaty'],
    'wordnet_code_of_conduct_105668095': ['Code_of_Conduct'],
    'wordnet_decree_106539770': ['Decree'],
    'wordnet_written_agreement_106771653': ['Decree'],
    'wordnet_criminal_record_106490173': ['Decree'],
    'wordnet_mandate_106556481': ['Decree'],
    'wikicat_Legal_doctrines_and_principles': ['Legal_Doctrine'],
    'wordnet_case_law_106535035': ['Precedent'],
    'wordnet_common_law_108453722': ['Customary_Law'],
    'wordnet_proclamation_101266491': ['Proclamation'],
    'wordnet_right_105174653': ['Right'],
    'wordnet_obligation_106773150': ['Obligation'],
    'wordnet_legislature_108163273': ['Legislative_Body'],
    'wordnet_court_108329453': ['Legislative_Body'],
    'wordnet_association_108049401': ['Society'],
    'wordnet_cooperative_101100877': ['Co-operative'],
    'wordnet_company_108058098': ['Company'],
    'wordnet_limited_company_108185211': ['Limited_Company'],
    'wordnet_corporation_108059412': ['Corporation'],
    'wordnet_foundation_108406486': ['Foundation'],
    'wordnet_pleading_106559365': ['Legal_Speech_Act'],
    'wordnet_party_110402824': ['Legal_Role'],
    'wordnet_judge_110225219': ['Professional_Legal_Role'],
    'wordnet_judiciary_108166187': ['Professional_Legal_Role'],
    'wordnet_lawyer_110249950': ['Professional_Legal_Role'],
}


def add_entities_from_list(pair_list, class_name, current_result):
    """Adds the entities of pair_list and their classes to current_result.

    The format of the pair_list is the same as the result of the script
    dowload_class_ids"""
    for entity, _ in pair_list[1:]:  # Ignore the first entry
        current_result[entity].update(YAGO_TO_LKIF_MAPPING[class_name])
    print 'Adding %d entities for class %s' % (len(pair_list), class_name)


def add_entities_from_directory(input_dirname, current_result):
    """Add entities from all the files from the directory to current_result.
    """
    # Read files
    filenames = [f for f in listdir(input_dirname)
                 if isfile(join(input_dirname, f))]
    for filename in filenames:
        if not ('.' in filename and filename.split('.')[1] == 'pickle'):
            # Not a pickle file
            continue
        class_name = filename[:-7].split('-')[0]
        if not class_name in YAGO_TO_LKIF_MAPPING:
            print 'Class %s not in mapping' % class_name
            continue
        with open(join(input_dirname, filename), 'r') as input_file:
            add_entities_from_list(pickle.load(input_file), class_name,
                                   current_result)


def main(input_dirnames, output_filename):
    result = defaultdict(lambda: set())

    for dirname in input_dirnames:
        add_entities_from_directory(dirname, result)

    with open(output_filename, 'w') as output_file:
        pickle.dump(dict(result), output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dirnames', type=unicode, nargs='+')
    parser.add_argument('-o', '--output_filename', type=unicode)

    args = parser.parse_args()

    main(args.input_dirnames, args.output_filename)
