# lmutils.r

## Table of Contents
[Installation](#installation)
[Important Information](#important)
[Matrix Functions](#matrix)
[Data Frame Functions](#data)
[Configuration](#configuration)

## Installation

Requires the [Rust programming language](https://rust-lang.org).

```sh 
# select option 1, default installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then install the package using the following command:

```r
install.packages("https://github.com/mrvillage/lmutils.r/archive/refs/heads/master.tar.gz", repos=NULL) # use .zip for Windows
# OR
devtools::install_github("mrvillage/lmutils.r")
```

## Important Information

### Terms
- Matrix convertable object - a data frame, matrix, file name (to read from), or column vector.
- List of matrix convertable objects - a list of matrix convertable objects, a character vector of file names (to read from), or a single matrix convertable object.
- Standard non-mutating output - a character vector of file names matching the length of the inputs, or `NULL` to return the output. If a single input, not in a list, was provided, the output will not be in a list.

### File Types
- `csv` (requires column headers)
- `tsv` (requires column headers)
- `txt` (requires column headers)
- `json`
- `cbor`
- `rkyv`
- `rdata` (NOTE: these files can only be processed sequentially, not in parallel like the rest)
All files can be optionally compressed with `gzip`, `rdata` files are assumed to be compressed without looking for a `.gz` file extension.

## Matrix Functions

### `lmutils::save`

Saves a list of matrix convertable objects to files.
- `from` is a list of matrix convertable objects.
- `to` is a character vector of file names to write to.

```r
lmutils::save(
    list("file1.csv", matrix(1:9, nrow=3), 1:3, data.frame(a=1:3, b=4:6)),
    c("file1.json", "file2.rkyv.gz", "file3.csv", "file4.rdata"),
)
```

### `lmutils::save_dir`

Recursively converts a directory of files to the selected file type.
- `from` is a string directory name to read the files from.
- `to` is a string directory name to write the files to or `NULL` to write to `from`.
- `file_type` is a string file extension to write the files as.

```r
lmutils::save_dir(
    "data",
    "converted_data", # or NULL
    "rkyv.gz",
)
```

### `lmutils::calculate_r2`

Calculates the R^2 and adjusted R^2 values for blocks and outcomes.
- `data` is a list of matrix convertable objects.
- `outcomes` is a single matrix convertable object.
Returns a data frame with columns `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted`.

```r
results <- lmutils::calculate_r2(
    c("block1.csv", "block2.rkyv.gz"),
    "outcomes1.RData",
)
```

### `lmutils::column_p_values`

Compute the p value of a linear regression between each pair of columns in data and outcomes.
- `data` is a list of matrix convertable objects.
- `outcomes` is a single matrix convertable object.
The function returns a data frame with columns `p_value`, `data`, `data_column`, and `outcome`.

```r
results <- lmutils::column_p_values(
    c("block1.csv", "block2.rkyv.gz"),
    "outcomes1.RData",
)
```

### `lmutils::combine_vectors`

Combine a list of double vectors into a single matrix using the vectors as columns.
- `data` is a list of double vectors.
- `out` is an output file name or `NULL` to return the matrix.

```r
lmutils::combine_vectors(
    list(1:3, 4:6),
    "combined_matrix.csv",
)
```

### `lmutils::remove_rows`

Removes rows from a matrix.
- `data` is list of matrix convertable objects.
- `rows` is a vector of row indices (1-based) to remove.
- `out` is a standard non-mutating output.

```r
lmutils::remove_rows(
    "matrix1.csv",
    c(1, 2, 3),
    "matrix1_removed_rows.csv",
)
```

### `lmutils::crossprod`

Calculates the cross product of two matrices. Equivalent to `t(data) %*% data`.
- `data` is a list of matrix convertable objects.
- `out` is a standard non-mutating output.

```r
lmutils::crossprod(
    "matrix1.csv",
    "crossprod_matrix1.csv",
)
```

### `lmutils::mul`

Multiplies two matrices. Equivalent to `a %*% b`.
- `a` is a list of matrix convertable objects.
- `b` is a list of matrix convertable objects.
- `out` is a standard non-mutating output.

```r
lmutils::mul(
    "matrix1.csv",
    "matrix2.rkyv.gz",
    "mul_matrix1_matrix2.csv",
)
```

### `lmutils::load`

Loads a matrix convertable object into R.
- `obj` is a list matrix convertable objects.
If a single object is provided, the function will return the matrix directly, otherwise it will return a list of matrices.

```r
lmutils::load("matrix1.csv")
```

### `lmutils::match_rows`

Matches rows of a matrix by the values of a vector.
- `data` is a list of matrix convertable objects.
- `with` is a numeric vector.
- `by` is the column name to match the rows by.
- `out` is a standard non-mutating output.

```r
lmutils::match_rows(
    "matrix1.csv",
    c(1, 2, 3),
    "eid",
    "matched_matrix1.csv",
)
```

### `lmutils::match_rows_dir`

Matches rows of all matrices in a directory to the values in a vector by a column.
- `from` is a string directory name to read the files from.
- `to` is a string directory name to write the files to or `NULL` to write to `from`.
- `with` is a numeric vector to match the rows to.
- `by` is the column name to match the rows by.

```r
lmutils::match_rows_dir(
    "matrices",
    "matched_matrices",
    c(1, 2, 3),
    "eid",
)
```

### `lmutils::dedup`

Deduplicate a matrix by a column. The first occurrence of each value is kept.
- `data` is a list of matrix convertable objects.
- `by` is the column name to deduplicate by.
- `out` is a standard non-mutating output.

```r
lmutils::dedup(
    "matrix1.csv",
    "eid",
    "matrix1_dedup.csv",
)
```

## Data Frame Functions

### `lmutils::new_column_from_regex`

Compute a new column for a data frame from a [Rust-flavored regex](https://docs.rs/regex/latest/regex/#syntax) and an existing column.
- `df` is a data frame.
- `column` is the column name to match.
- `regex` is the regex to match. The first capture group is used.
- `new_column` is the new column name.

```r
lmutils::new_column_from_regex(
    data.frame(a=c("a1", "b2", "c3")),
    "a",
    "([a-z])",
    "b",
)
```

### `lmutils::map_from_pairs`

Converts two character vectors into a named list, where the first vector is the names and the second vector is the values. Only the first occurrence of each name is used, essentially creating a map.
- `names` is a character vector of names.
- `values` is a character vector of values.

```r
lmutils::map_from_pairs(
    c("a", "b", "c"),
    c("1", "2", "3"),
)
```

### `lmutils::new_column_from_map`

Compute a new column for a data frame from a list of values and an existing column, matching by the names of the values.
- `df` is a data frame.
- `column` is the column name to match.
- `values` is a named list of values.
- `new_column` is the new column name.

```r
lmutils::new_column_from_map(
    data.frame(a=c("a", "b", "c")),
    "a",
    lmutils::map_from_pairs(
        c("a", "b", "c"),
        c("1", "2", "3"),
    ),
    "b",
)
```

### `lmutils::new_column_from_map_pairs`

Compute a new column for a data frame from two character vectors of names and values, matching by the names.
- `df` is a data frame.
- `column` is the column name to match.
- `names` is a character vector of names.
- `values` is a character vector of values.
- `new_column` is the new column name.

```r
lmutils::new_column_from_map_pairs(
    data.frame(a=c("a", "b", "c")),
    "a",
    c("a", "b", "c"),
    c("1", "2", "3"),
    "b",
)
```

### `lmutils::df_sort_asc`

Mutably sorts a data frame in ascending order by multiple columns in ascending order. All columns must be numeric (double or integer), character, or logical vectors.
- `df` is a data frame.
- `columns` is a character vector of column names to sort by.

```r
df <- data.frame(a=c(3, 3, 2, 2, 1, 1), b=c("b", "a", "b", "a", "b", "a"))
lmutils::df_sort_asc(
    df,
    c("a", "b"),
)
```

## Configuration

`lmutils` exposes three global config options that can be set using environment variables or the `lmutils` package functions:

- `LMUTILS_LOG`/`lmutils::set_log_level` to set the log level (default: `info`). Available log levels in order of increasing verbosity are `off`, `error`, `warn`, `info`, `debug`, and `trace`.
- `LMUTILS_CORE_PARALLELISM`/`lmutils::set_core_parallelism` to set the core parallelism (default: `16`). This is the number of primary operations to run in parallel.
- `LMUTILS_NUM_WORKER_THREADS`/`lmutils::set_num_worker_threads` to set the number of worker threads to use (default: `num_cpus::get() / 2`). This is the number of threads to use for parallel operations. Once an operation has been run, this value cannot be changed.
