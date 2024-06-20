# lmutils.r

## Installation

Requires the [Rust programming language](https://rust-lang.org).

```r
devtools::install_github("mrvillage/lmutils.r")
# OR
install.packages("https://github.com/mrvillage/lmutils.r/archive/refs/heads/master.tar.gz", repos=NULL) # use .zip for Windows
```

## Important Notes

- RData files **CANNOT** be read in parallel like the other formats. It is **HIGHLY** recommended to convert RData files to another format using `lmutils::convert_file` before processing them. The fastest and smallest format is `rkyv.gz`.
- RData files are assumed to be compressed without looking for a `.gz` file extension.
- All files are looked for in the current working directory.
- All files are written to the current working directory.
- All files are assumed to be matrices of floats, unless otherwise specified.

## Functions

### `lmutils::convert_file`

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
lmutils::convert_file(
    c("file1.csv", "file2.RData"),
    c("file1.json", "file2.rkyv.gz"),
)
```

### `lmutils::calculate_r2`

Calculates the R^2 and adjusted R^2 values for blocks and outcomes.

The first argument is a character vector of file names to read the blocks from, a list of matrices to use as the blocks, or a single matrix.

The second argument is a single file name or matrix to use as the outcomes. Each outcome is a column in the matrix.

The function returns a data frame with columns `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted`.

```r
results <- lmutils::calculate_r2(
    c("block1.csv", "block2.rkyv.gz"),
    "outcomes1.RData",
)
```

### `lmutils::calculate_r2_ranges`

Calculates the R^2 and adjusted R^2 values for blocks and outcomes for a range of columns. Each block is a one range of columns in the provided matrix.

**You will likely never need this function `lmutils::calculate_r2` is much more useful.**

The first argument is file name to read the matrix from or a matrix.

The second argument is a single file name or matrix to use as the outcomes. Each outcome is a column in the matrix.

The third argument is a matrix with two columns, the start and end columns to use (inclusive).

The function returns a data frame with columns `r2`, `adj_r2`, and `outcome` corresponding to each range in order.

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


### `lmutils::save_matrix`

Saves a matrix to a file.

The first argument is a matrix to save.

The second argument is a string file name to save the matrix to.

```r
lmutils::save_matrix(
    matrix(1:9, nrow=3),
    "matrix1.csv",
)
```

### `lmutils::to_matrix`

Converts a numeric data frame, numeric matrix, or RData file to a matrix.

The first argument is the data frame, matrix, or file name to convert.

The second argument is a string file name to save the matrix to.

If the second argument is `NULL`, the function will return the matrix.

```r
lmutils::to_matrix(
    data.frame(a=1:3, b=4:6),
    "matrix1.csv",
)
lmutils::to_matrix(
    matrix(1:9, nrow=3),
    "matrix2.csv",
)
lmutils::to_matrix(
    "matrix1.RData",
    "matrix3.csv",
)
```

### `lmutils::crossprod`

Calculates the cross product of two matrices. Equivalent to `t(data) %*% data`.

The first argument is a string file name or a matrix to read the matrix from.

The second argument is a string file name to save the matrix to.

If the second argument is `NULL`, the function will return the cross product matrix.

```r
lmutils::crossprod(
    "matrix1.csv",
    "crossprod_matrix1.csv",
)
```

### `lmutils::to_matrix_dir`

Recursively converts all files in a directory to matrices.

The first argument is a string directory name to read the files from.

The second argument is a string directory name to write the matrices to.

The third argument is a string file extension to write the matrices as.

If the second argument is `NULL`, the matrices will be written to the input directory.

```r
lmutils::to_matrix_dir(
    "data",
    "matrices",
    "csv.gz",
)
```

### `lmutils::standardize`

Standardize a matrix. All NaN values are replaced with the mean of the column and each column is scaled to have a mean of 0 and a standard deviation of 1.

The first argument is a string file name or a matrix to read the matrix from.

The second argument is a string file name to save the matrix to.

If the second argument is `NULL`, the function will return the standardized matrix.

```r
lmutils::standardize(
    "matrix1.csv",
    "standardized_matrix1.csv",
)
```


### `lmutils::column_p_values`

Computes the p-values of a linear regression between each pair of columns in two matrices.

The first argument is a character vector of file names to read from, a list of matrices to use as the blocks, or a single matrix.

The second argument is a string file name or a matrix to read from.

The function returns a data frame with columns `p_value`, `data`, `data_column`, and `outcome` corresponding to each range in order.

```r
results <- lmutils::column_p_values(
    c("block1.csv", "block2.rkyv.gz"),
    "outcomes1.RData",
)
```

## Configuration

`lmutils` exposes three global config options that can be set using environment variables or the `lmutils` package functions:

- `LMUTILS_LOG`/`lmutils::set_log_level` to set the log level (default: `info`).
- `LMUTILS_NUM_MAIN_THREADS`/`lmutils::set_num_main_threads` to set the number of main threads to use (default: `16`). This is the number of primary operations to run in parallel.
- `LMUTILS_NUM_WORKER_THREADS`/`lmutils::set_num_worker_threads` to set the number of worker threads to use (default: `num_cpus::get() / 2`). This is the number of threads to use for parallel operations. Once an operation has been run, this value cannot be changed.
