#!/bin/bash

set -euo pipefail

result=0

for var in TENSORFLOW_PACKAGE KERAS_PACKAGE PYTORCH_PACKAGE PYTORCH_LIGHTNING_PACKAGE TORCHVISION_PACKAGE MXNET_PACKAGE PYSPARK_PACKAGE
do
  if [ -z "${!var-}" ]
  then
    echo "environment var $var not set"
    result=1
    continue
  fi

  pattern="${!var}"
  if [ "$pattern" == "None" ]
  then
    continue
  elif [[ "$pattern" == "tf-nightly" ]] ||
       [[ "$pattern" == "tf-nightly-gpu" ]]
  then
    flag="-P"
    pattern="$pattern==.*\\.dev20\\d{6}"
  elif [[ "$pattern" == "torch-nightly"* ]] ||
       [[ "$pattern" == "torchvision" ]]
  then
    flag="-P"
    pattern="${pattern/-nightly*/}==.*\\.dev20\\d{6}.*"
  elif [[ "$pattern" == "mxnet-nightly"* ]]
  then
    flag="-P"
    pattern="${pattern/-nightly/}==.*20\\d{6}"
  else
    flag="-Fx"
  fi

  found=$(pip freeze | grep -i $flag "$pattern" || true)
  if [ -n "$found" ]
  then
    if [ "$found" == "$pattern" ]
    then
      echo "$found found"
    else
      echo "$found found (matches ${!var} / $pattern)"
    fi
  else
    found=$(pip freeze | grep -i "^${pattern/=*/==}" || true)
    if [ -n "$found" ]
    then
      echo "$found found BUT ${!var} / $pattern expected"
    else
      echo "$pattern NOT found (no match for ${!var} / $pattern)"
    fi
    result=1
  fi
done

exit $result

