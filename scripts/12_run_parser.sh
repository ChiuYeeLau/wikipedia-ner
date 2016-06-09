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
  echo "Analyzing $filename"
  java -Xmx3G -jar ~/opt/maltparser-1.8.1/maltparser-1.8.1.jar -w ~/opt/maltparser-1.8.1/models/ -c engmalt.linear-1.7 -m parse -i $file -o /tmp/$filename &> logs/$filename &
done
wait)

find /tmp -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=$(basename $file)
  echo "Simplyfing $filename"
  awk '{ print $1 "\t" $2 "\t" $4 "\t" $6 "\t" $7 "\t" $8 }' /tmp/$filename > $2/$filename &
  rm -f /tmp/$filename
done
wait)
