#!/usr/bin/env bash

function maxjobs {
  while [ $(jobs | wc -l) -ge 4 ]
  do
    sleep 5
  done
}


find ~/wikipedia/resources/docs_for_ner_parsed -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=$(basename $file)
  echo "Analyzing $filename"
  awk '{ print $1 "\t" $2 "\t" $4 "\t" $6 "\t" $7 "\t" $8 }' $file > ~/wikipedia/resources/docs_for_ner/$filename &
done
wait)
