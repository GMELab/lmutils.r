#/usr/bin/bash
dir=$1
jobs=$2

cd $( dirname -- "${BASH_SOURCE[0]}" )

find $dir -type f | parallel -j $jobs Rscript ./df_to_rkyv.r
