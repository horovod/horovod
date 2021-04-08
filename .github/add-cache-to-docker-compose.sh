#!/bin/bash

set -euo pipefail

repository="$1"

IFS=''
test=''
build=0
while read line
do
  # add cache_from to earlier before this line
  if [[ $line == "  test-"* ]] || [[ $build -eq 1 ]] && [[ $line != "     "* ]]
  then
    if [[ -n "$test" ]] && [[ "$test" != *"-base:" ]]
    then
      if [[ $build -eq 0 ]]
      then
        echo "    build:"
      fi
      echo "      cache_from:"
      echo "        - ${repository}:horovod-${test/%:/}-latest"
      test=""
    fi
    if [[ $line == "  test-"* ]]
    then
      test=${line/#  /}
      build=0
    fi
  fi

  # detect if current test has a build section
  if [[ $line == "    build:" ]]
  then
    build=1
  fi
  echo "$line"
done
echo "      cache_from:"
echo "        - ${repository}:horovod-${test/%:/}-latest"

