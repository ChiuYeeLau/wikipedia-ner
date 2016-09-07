#!/usr/bin/env perl

use strict;
use warnings;
use File::Basename;

my $models_dir = shift @ARGV;
my $evaluation_dataset = shift @ARGV;
my $classes_file = shift @ARGV;
my $evaluation_words = shift @ARGV;
my $results_dir = shift @ARGV;
my $word_vectors = shift @ARGV;

my @models = `find $models_dir -type f -name "*NEU*.model"`;
chomp @models;

print "#!/usr/bin/env bash\nset -e\n\n";

foreach my $model(@models) {
    my ($filename, $dirname, undef) = fileparse($model, qr/\.model/);
    my $experiment_name = basename($dirname);

    $filename =~ m/[A-Z_]*([0-9_]*)/;
    my $layers = join(" ", split("_", $1));

    print "echo \"Running experiment $experiment_name\"\n";
    my $cmd = "python evaluate_corpus.py $evaluation_dataset $classes_file $model ";
    $cmd = $cmd . "$evaluation_words $results_dir/$experiment_name.txt --layers $layers ";
    $cmd = $cmd . "$word_vectors\n";

    print $cmd;
    print "echo \"Finished experiment $experiment_name\"\n\n";
}

print "echo \"All experiments finished\"\n";
