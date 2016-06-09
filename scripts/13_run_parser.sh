#!/usr/bin/env bash

function maxjobs {
  while [ `jobs | wc -l` -ge 4 ]
  do
    sleep 5
  done
}

mkdir -p ../resources/docs_for_ner_parsed

find ../resources/docs_for_ner_tagged -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=${file#../resources/docs_for_ner_tagged/}
  echo "Analyzing $filename"
  grep -v "^#" $file > /tmp/${filename}_tmp 
  rm -f $file
  java -Xmx3G -jar ~/opt/maltparser-1.8.1/maltparser-1.8.1.jar -w ~/opt/maltparser-1.8.1/models/ -c engmalt.linear-1.7 -m parse -i /tmp/${filename}_tmp -o ../resources/docs_for_ner_parsed/$filename &> logs/$filename &
done
wait)
