#!/bin/bash

set -euo pipefail

result=0

for var in TENSORFLOW_PACKAGE KERAS_PACKAGE PYTORCH_PACKAGE PYTORCH_LIGHTNING_PACKAGE TORCHVISION_PACKAGE MXNET_PACKAGE PYSPARK_PACKAGE
do
  if [ -z "${!var-}" ]
  then
    echo -e "$var \u001b[31mnot set\u001b[0m"
    result=1
    continue
  fi

  pattern="${!var}"
  if [ "$pattern" == "None" ]
  then
    continue
  elif [[ "$pattern" == "tf-nightly" ]]
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
      echo -e "$found \u001b[32mfound\u001b[0m"
    else
      echo -e "$found \u001b[32mfound\u001b[0m (matches ${!var} / $pattern)"
    fi
  else
    found=$(pip freeze | grep -i "^${pattern/=*/==}" || true)
    if [ -n "$found" ]
    then
      echo -e "$found \u001b[31mfound BUT\u001b[0m ${!var} / $pattern \u001b[31mexpected\u001b[0m"
    else
      echo -e "$pattern \u001b[31mNOT found\u001b[0m (no match for ${!var} / $pattern)"
    fi
    result=1
  fi
done

exit $result

