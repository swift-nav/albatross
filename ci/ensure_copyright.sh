#!/bin/bash

has_swift_copyright() {
  SECOND_LINE=`sed -n 2p $1`
  if [[ "$SECOND_LINE" = *"Swift"* ]]; then
    return
  fi
  false
}

THIS_DIRECTORY=`dirname $0`
cd $THIS_DIRECTORY

for f in `find ../include/albatross ../tests ../examples -type f \( -iname \*.h -o -iname \*.cc -o -iname \*.hpp \)`; do
  if ! has_swift_copyright $f; then
    echo "bad copyright header in $f"
    exit -1
  fi
  echo "has copyright: $f";
done

