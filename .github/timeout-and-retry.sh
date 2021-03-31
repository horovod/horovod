#!/bin/bash

set -euo pipefail

limit=$1
attempts=$2
wait=$3
shift 3

attempt=1
while ! timeout ${limit} "$@"
do
  echo "Attempt $((attempt++)) timed out!"
  if [ $attempt -ge $attempts ]
  then
    exit 1
  fi
  sleep $wait

  echo
  echo "Retry"
done
