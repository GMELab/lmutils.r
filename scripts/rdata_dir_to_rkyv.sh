#/usr/bin/bash
dir=$1
jobs=$2

if [ "$#" -ne 2 ]; then
    echo "Usage: rdata_dir_to_rkyv.sh <dir> <jobs>"
    exit 1
fi

cd $( dirname -- "${BASH_SOURCE[0]}" )

find $dir -type f | parallel -j $jobs Rscript ./df_to_rkyv.r
