#!/usr/bin/env perl

use strict;
use warnings;

my @files = `find ../../resources/movies/docs_for_ner -type f -name "*.conll"`;
chomp @files;
my %sentences = ();

foreach my $file(@files) {
    print STDERR "Counting sentences for $file";
    $sentences{$file} = `grep -E '^\\s*\$' $file | wc -l`;
}

print STDERR "Counting finished";

open(my $fh, ">", "../../resources/movies/conll_sentences.txt");

foreach my $key (sort keys %sentences) {
    print $fh "$key: " . $sentences{$key} . "\n";
}

close $fh;
