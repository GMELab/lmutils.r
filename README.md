# lmutils.r

## Table of Contents
[Installation](#installation)
[Important Information](#important)
[Mat Objects](#mat)
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
- Matrix convertable object - a data frame, matrix, file name (to read from), a numeric column vector, or a `Mat` object.
- List of matrix convertable objects - a list of matrix convertable objects, a character vector of file names (to read from), or a single matrix convertable object.
- Standard output file - a character vector of file names matching the length of the inputs, or `NULL` to return the output. If a single input, not in a list, was provided, the output will not be in a list.

### File Types
- `csv` (requires column headers)
- `tsv` (requires column headers)
- `txt` (requires column headers)
- `json`
- `cbor`
- `rkyv`
- `rdata` (NOTE: these files can only be processed sequentially, not in parallel like the rest)
All files can be optionally compressed with `gzip`, `rdata` files are assumed to be compressed without looking for a `.gz` file extension.

## `Mat` Objects

`lmutils::Mat` objects are a way to store matrices in memory and perform operations on them. They can be used to store operations or chain operations together for later execution. This can be useful if, for example, you wish to a hundred large matrices from files and standardize them all before using `lmutils::calculate_r2`. Using `Mat` objects, you can store the operations you wish to perform and `Mat` will execute them only when the matrix is loaded.

### `lmutils::Mat$new`

Creates a new `Mat` object.
- `data` is a matrix convertable object.

```r
mat <- lmutils::Mat$new("matrix1.csv")
```

### `lmutils::Mat$r`

Loads the matrix from the `Mat` object.

```r
m <- mat$r()
```

### `lmutils::Mat$save`

Saves the matrix to a file.
- `file` is the file name to write to.

```r
mat$save("matrix1.rkyv.gz")
```

### `lmutils::Mat$combine_columns`

Combines this matrix with other matrices by columns. (`cbind`)
- `data` is a list of matrix convertable objects.

```r
mat$combine_columns("matrix2.csv")
```

### `lmutils::Mat$combine_rows`

Combines this matrix with other matrices by rows. (`rbind`)
- `data` is a list of matrix convertable objects.

```r
mat$combine_rows("matrix2.csv")
```

### `lmutils::Mat$remove_columns`

Removes columns from the matrix.
- `columns` is a vector of column indices (1-based) to remove.

```r
mat$remove_columns(c(1, 2, 3))
```

### `lmutils::Mat$remove_column`

Removes a column from the matrix by name.
- `column` is the column name to remove.

```r
mat$remove_column("eid")
```

### `lmutils::Mat$remove_column_if_exists`

Removes a column from the matrix by name if it exists.
- `column` is the column name to remove.

```r
mat$remove_column_if_exists("eid")
```

### `lmutils::Mat$remove_rows`

Removes rows from the matrix.
- `rows` is a vector of row indices (1-based) to remove.

```r
mat$remove_rows(c(1, 2, 3))
```

### `lmutils::Mat$transpose`

Transposes the matrix.

```r
mat$transpose()
```

### `lmutils::Mat$sort`

Sort by the column at the given index.
- `by` is the column index (1-based) to sort by.

```r
mat$sort(1)
```

### `lmutils::Mat$sort_by_name`

Sort by the column with the given name.
- `by` is the column name to sort by.

```r
mat$sort_by_name("eid")
```

### `lmutils::Mat$sort_by_order`

Sort by the given order of rows.
- `order` is a vector of row indices (1-based) to sort by.

```r
mat$sort_by_order(c(3, 2, 1))
```

### `lmutils::Mat$dedup`

Deduplicate the matrix by a column.
- `by` is the column index (1-based) to deduplicate by.

```r
mat$dedup(1)
```

### `lmutils::Mat$dedup_by_name`

Deduplicate the matrix by a column name.
- `by` is the column name to deduplicate by.

```r
mat$dedup_by_name("eid")
```

### `lmutils::Mat$match_to`

Match the rows of the matrix to the values in a vector by a column.
- `with` is a numeric vector to match the rows to.
- `by` is the column index (1-based) to match the rows by.
- `join` is the type of join to perform. 0 is inner, 1 is left, 2 is right, and 3 is full. If a row is not matched for a left or right join, it will error.

```r
mat$match_to(c(1, 2, 3), 1, 0)
```

### `lmutils::Mat$match_to_by_name`

Match the rows of the matrix to the values in a vector by a column name.
- `with` is a numeric vector to match the rows to.
- `by` is the column name to match the rows by.
- `join` is the type of join to perform. 0 is inner, 1 is left, 2 is right, and 3 is full. If a row is not matched for a left or right join, it will error.

```r
mat$match_to_by_name(c(1, 2, 3), "eid", 0)
```

### `lmutils::Mat$join`

Join the matrix with another matrix by a column.
- `other` is a matrix convertable object.
- `by` is the column index (1-based) to join by.
- `join` is the type of join to perform. 0 is inner, 1 is left, 2 is right, and 3 is full. If a row is not matched for a left or right join, it will error.

```r
mat$join("matrix2.csv", 1, 0)
```

### `lmutils::Mat$join_by_name`

Join the matrix with another matrix by a column name.
- `other` is a matrix convertable object.
- `by` is the column name to join by.
- `join` is the type of join to perform. 0 is inner, 1 is left, 2 is right, and 3 is full. If a row is not matched for a left or right join, it will error.

```r
mat$join_by_name("matrix2.csv", "eid", 0)
```

### `lmutils::Mat$standardize_columns`

Standardize the columns of the matrix to have a mean of 0 and a standard deviation of 1.

```r
mat$standardize_columns()
```

### `lmutils::Mat$standardize_rows`

Standardize the rows of the matrix to have a mean of 0 and a standard deviation of 1.

```r
mat$standardize_rows()
```

### `lmutils::Mat$remove_na_rows`

Remove rows with any `NA` values.

```r
mat$remove_na_rows()
```

### `lmutils::Mat$remove_na_columns`

Remove columns with any `NA` values.

```r
mat$remove_na_columns()
```

### `lmutils::Mat$na_to_value`

Replace all `NA` values with a given value.

```r
mat$na_to_value(0)
```

### `lmutils::Mat$na_to_column_mean`

Replace all `NA` values with the mean of the column.

```r
mat$na_to_column_mean()
```

### `lmutils::Mat$na_to_row_mean`

Replace all `NA` values with the mean of the row.

```r
mat$na_to_row_mean()
```

### `lmutils::Mat$min_column_sum`

Remove columns with a sum less than a given value.

```r
mat$min_column_sum(10)
```

### `lmutils::Mat$max_column_sum`

Remove columns with a sum greater than a given value.

```r
mat$max_column_sum(10)
```

### `lmutils::Mat$min_row_sum`

Remove rows with a sum less than a given value.

```r
mat$min_row_sum(10)
```

### `lmutils::Mat$max_row_sum`

Remove rows with a sum greater than a given value.

```r
mat$max_row_sum(10)
```

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
- `out` is a standard output file.

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
- `out` is a standard output file.

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
- `out` is a standard output file.

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
- `out` is a standard output file.

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
- `out` is a standard output file.

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
