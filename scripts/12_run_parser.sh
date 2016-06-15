#!/usr/bin/env bash

function maxjobs {
  while [ `jobs | wc -l` -ge 4 ]
  do
    sleep 5
  done
}

find $1 -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=$(basename $file)
  [[ $filename == "doc_03.conll" ]] && continue
  echo "Analyzing $filename"
  java -Xmx3G -jar ~/opt/maltparser-1.8.1/maltparser-1.8.1.jar -w ~/opt/maltparser-1.8.1/models/ -c engmalt.linear-1.7 -m parse -i $file -o $2/$filename &> logs/${filename}.parser.log &
done
wait)

find $2 -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=$(basename $file)
  [[ $filename == "doc_03.conll" ]] && continue
  echo "Simplyfing $filename"
  awk '{ print $1 "\t" $2 "\t" $4 "\t" $6 "\t" $7 "\t" $8 }' $file > $3/$filename &
done
wait)

echo "All jobs done"
