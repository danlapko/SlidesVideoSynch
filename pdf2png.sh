#!/bin/bash
# splits pdf with slides into slides pictures
# $1 = pdf file, $2 = dir_to_store_result

# imagemagick must be installed
convert -density 300 "$1" -quality 100 "$2"