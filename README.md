# lmutils.r

## Installation

```r
devtools::install_github("mrvillage/lmutils.r")
```

## Important Notes

- RData files **CANNOT** be read in parallel like the other formats. It is **HIGHLY** recommended to convert RData files to another format using `lmutils::convert_files` before processing them. The fastest and smallest format is `rkyv.gz`.
- RData files are assumed to be compressed without looking for a `.gz` file extension.
- All files are looked for in the current working directory.
- All files are written to the current working directory.
- All files are assumed to be matrices of floats, unless otherwise specified.

## Functions

### `lmutils::convert_files`

Converts matrix files from one format to another. Supported formats are:
- `csv` (requires column headers)
- `tsv` (requires column headers)
- `txt` (requires column headers)
- `json`
- `cbor`
- `rkyv`
- `rdata` (NOTE: these files can only be processed sequentially, not in parallel like the rest)

All files can be optionally compressed with `gzip`, `rdata` files are assumed to be compressed without looking for a `.gz` file extension.

```r
lmutils::convert_files(
    c("file1.csv", "file2.RData"),
    c("file1.json", "file2.rkyv.gz"),
    0 # 0 means read as a matrix of floats, 1 means read as a matrix of strings
)
```

### `lmutils::calculate_r2`

Calculates the R^2 and adjusted R^2 values for a block and outcomes.

The first argument is a character vector of file names to read the blocks from, a list of matrices to use as the blocks, or a single matrix.

The second argument is a single file name or matrix to use as the outcomes. Each outcome is a column in the matrix.

The function returns a data frame with columns `r2` and `adj_r2` corresponding to each outcome for each block in order.

```r
results <- lmutils::calculate_r2(
    c("block1.csv", "block2.rkyv.gz"),
    "outcomes1.RData",
)
```

### `lmutils::calculate_r2_ranges`

Calculates the R^2 and adjusted R^2 values for blocks and outcomes for a range of columns. Each block is a one range of columns in the provided matrix.

The first argument is file name to read the matrix from or a matrix.

The second argument is a single file name or matrix to use as the outcomes. Each outcome is a column in the matrix.

The third argument is a matrix with two columns, the start and end columns to use (inclusive).

The function returns a data frame with columns `r2` and `adj_r2` corresponding to each outcome for each block in order.

```r
results <- lmutils::calculate_r2_ranges(
    "blocks1.csv",
    "outcomes1.RData",
    matrix(c(1, 10, 11, 20), ncol=2),
)
```

### `lmutils::combine_matrices`

Combines matrices into a single matrix. The matrices must have the same number of rows.

The first argument is a character vector of file names to read the matrices from or a list of matrices.

The second argument is a file name to write the combined matrix to.

If the second argument is `NULL`, the function will return the combined matrix.

```r
lmutils::combine_matrices(
    c("matrix1.csv", "matrix2.rkyv.gz"),
    "combined_matrix.rkyv.gz",
)
```

### `lmutils::remove_rows`

Removes rows from a matrix.

The first argument is a string file name or a matrix to remove rows from.

The second argument is a vector of row indices to remove.

The third argument is a string file name to write the new matrix to.

If the third argument is `NULL`, the function will return the new matrix.

```r
lmutils::remove_rows(
    "matrix1.csv",
    c(1, 2, 3),
    "matrix1_removed_rows.csv",
)
```
