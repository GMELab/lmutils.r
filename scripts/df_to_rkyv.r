#!/usr/bin/Rscript
args <- (commandArgs(TRUE))
file <- args[1]

lmutils::to_matrix(file, paste0(sub(".RData", ".rkyv.gz", file)))
