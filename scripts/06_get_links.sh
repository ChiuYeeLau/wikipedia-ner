#!/usr/bin/env bash

function maxjobs {
  while [ `jobs | wc -l` -ge 6 ]
  do
    sleep 5
  done
}

find ../clean -type f -name "wiki_*" | (while read file
do
  maxjobs
  filename=${file#../clean/}
  echo "Analyzing $filename"
  grep -o -E '<doc[^>]*>|</doc>|<a[^>]*>[^<]*</a>' $file > ../links/$filename &
done
wait)
