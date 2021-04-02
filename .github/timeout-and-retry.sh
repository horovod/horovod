#!/bin/bash

set -euo pipefail

limit=$1
attempts=$2
wait=$3
shift 3

attempt=1
echo "::group::Starting attempt #$attempt: $*"
while ! timeout ${limit} "$@" 2>&1
do
  status=$?
  echo "::endgroup::"

  if [ $status == 124 ]
  then
    echo "::warning::Attempt #$attempt timed out: $*"
  else
    echo "::warning::Attempt #$attempt failed: $*"
  fi

  if [[ $((attempt++)) -ge $attempts ]]
  then
    echo "::error::$((attempts)) attempts failed or timed out, giving up"
    exit 1
  fi
  sleep $wait

  echo
  echo "::group::Starting attempt #$attempt: $*"
done
echo "::endgroup::"
