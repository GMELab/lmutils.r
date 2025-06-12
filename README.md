# lmutils.r

## Table of Contents
- [Installation](#installation)
- [Important Information](#important-information)
  - [Terms](#terms)
  - [File Types](#file-types)
- [Introduction](#introduction)
  - [Example](#example)
- [Mat Objects](#mat-objects)
  - [`lmutils::Mat$new`](#lmutilsmatnew)
  - [`lmutils::Mat$r`](#lmutilsmatr)
  - [`lmutils::Mat$col`](#lmutilsmatcol)
  - [`lmutils::Mat$colnames`](#lmutilsmatcolnames)
  - [`lmutils::Mat$save`](#lmutilsmatsave)
  - [`lmutils::Mat$combine_columns`](#lmutilsmatcombine_columns)
  - [`lmutils::Mat$combine_rows`](#lmutilsmatcombine_rows)
  - [`lmutils::Mat$remove_columns`](#lmutilsmatremove_columns)
  - [`lmutils::Mat$remove_column`](#lmutilsmatremove_column)
  - [`lmutils::Mat$remove_column_if_exists`](#lmutilsmatremove_column_if_exists)
  - [`lmutils::Mat$remove_rows`](#lmutilsmatremove_rows)
  - [`lmutils::Mat$transpose`](#lmutilsmattranspose)
  - [`lmutils::Mat$sort`](#lmutilsmatsort)
  - [`lmutils::Mat$sort_by_name`](#lmutilsmatsort_by_name)
  - [`lmutils::Mat$sort_by_order`](#lmutilsmatsort_by_order)
  - [`lmutils::Mat$dedup`](#lmutilsmatdedup)
  - [`lmutils::Mat$dedup_by_name`](#lmutilsmatdedup_by_name)
  - [`lmutils::Mat$match_to`](#lmutilsmatmatch_to)
  - [`lmutils::Mat$match_to_by_name`](#lmutilsmatmatch_to_by_name)
  - [`lmutils::Mat$join`](#lmutilsmatjoin)
  - [`lmutils::Mat$join_by_name`](#lmutilsmatjoin_by_name)
  - [`lmutils::Mat$standardize_columns`](#lmutilsmatstandardize_columns)
  - [`lmutils::Mat$standardize_rows`](#lmutilsmatstandardize_rows)
  - [`lmutils::Mat$remove_na_rows`](#lmutilsmatremove_na_rows)
  - [`lmutils::Mat$remove_na_columns`](#lmutilsmatremove_na_columns)
  - [`lmutils::Mat$na_to_value`](#lmutilsmatna_to_value)
  - [`lmutils::Mat$na_to_column_mean`](#lmutilsmatna_to_column_mean)
  - [`lmutils::Mat$na_to_row_mean`](#lmutilsmatna_to_row_mean)
  - [`lmutils::Mat$min_column_sum`](#lmutilsmatmin_column_sum)
  - [`lmutils::Mat$max_column_sum`](#lmutilsmatmax_column_sum)
  - [`lmutils::Mat$min_row_sum`](#lmutilsmatmin_row_sum)
  - [`lmutils::Mat$max_row_sum`](#lmutilsmatmax_row_sum)
  - [`lmutils::Mat$rename_column`](#lmutilsmatrename_column)
  - [`lmutils::Mat$rename_column_if_exists`](#lmutilsmatrename_column_if_exists)
  - [`lmutils::Mat$remove_duplicate_columns`](#lmutilsmatremove_duplicate_columns)
  - [`lmutils::Mat$remove_identical_columns`](#lmutilsmatremove_identical_columns)
  - [`lmutils::Mat$eigen`](#lmutilsmateigen)
  - [`lmutils::Mat$subset_columns`](#lmutilsmatsubset_columns)
  - [`lmutils::Mat$rename_columns_with_regex`](#lmutilsmatrename_columns_with_regex)
  - [`lmutils::Mat$scale_columns`](#lmutilsmatscale_columns)
- [Matrix Functions](#matrix-functions)
  - [`lmutils::save`](#lmutilssave)
  - [`lmutils::save_dir`](#lmutilssave_dir)
  - [`lmutils::calculate_r2`](#lmutilscalculate_r2)
  - [`lmutils::column_p_values`](#lmutilscolumn_p_values)
  - [`lmutils::linear_regression`](#lmutilslinear_regression)
  - [`lmutils::logistic_regression`](#lmutilslogistic_regression)
  - [`lmutils::combine_vectors`](#lmutilscombine_vectors)
  - [`lmutils::combine_rows`](#lmutilscombine_rows)
  - [`lmutils::remove_rows`](#lmutilsremove_rows)
  - [`lmutils::crossprod`](#lmutilscrossprod)
  - [`lmutils::mul`](#lmutilsmul)
  - [`lmutils::load`](#lmutilsload)
  - [`lmutils::match_rows`](#lmutilsmatch_rows)
  - [`lmutils::match_rows_dir`](#lmutilsmatch_rows_dir)
  - [`lmutils::dedup`](#lmutilsdedup)
- [Data Frame Functions](#data-frame-functions)
  - [`lmutils::new_column_from_regex`](#lmutilsnew_column_from_regex)
  - [`lmutils::map_from_pairs`](#lmutilsmap_from_pairs)
  - [`lmutils::new_column_from_map`](#lmutilsnew_column_from_map)
  - [`lmutils::new_column_from_map_pairs`](#lmutilsnew_column_from_map_pairs)
  - [`lmutils::df_sort_asc`](#lmutilsdf_sort_asc)
  - [`lmutils::df_split`](#lmutilsdf_split)
  - [`lmutils::df_combine`](#lmutilsnew_column_from_map_pairs)
- [Other Functions](#other-functions)
  - [`lmutils::compute_r2`](#lmutilscompute_r2)
  - [`lmutils::compute_r2_tjur`](#lmutilscompute_r2_tjur)
  - [`lmutils::mean`](#lmutilsmean)
  - [`lmutils::median`](#lmutilsmedian)
  - [`lmutils::sd`](#lmutilssd)
  - [`lmutils::var`](#lmutilsvar)
  - [`lmutils::num_cores`](#lmutilsnum_cores)
- [Configuration](#configuration)

## Installation

`lmutils` is not currently available on CRAN, but it can be installed on Linux with the following command. This will also install the [Rust programming language](https://rust-lang.org) which is required for `lmutils`.

```r
curl https://raw.githubusercontent.com/GMELab/lmutils.r/refs/heads/master/install.sh | sh
```

## Important Information

### Terms
- Matrix convertible object - a data frame, matrix, file name (to read from), a numeric column vector, or a `Mat` object.
- List of matrix convertible objects - a list of matrix convertible objects, a character vector of file names (to read from), or a single matrix convertible object.
- Standard output file - a character vector of file names matching the length of the inputs, or `NULL` to return the output. If a single input, not in a list, was provided, the output will not be in a list.
- Join - an inner join means only rows that match in both matrices are kept, a left join means all rows in the left matrix are kept, a right join means all rows in the right matrix are kept.

### File Types
All files can be optionally compressed with `gzip`, `rdata` files are assumed to be compressed without looking for a `.gz` file extension (as is the standard in R).
- `.mat` (recommended, custom binary format designed for matrices)
- `.csv` (requires column headers)
- `.tsv` (requires column headers)
- `.txt` (requires column headers)
- `.json`
- `.cbor`
- `.rkyv`
- `.rdata`
- `.rds`

## Introduction

`lmutils` is an R package that provides utilities for working with matrices and data frames. It is built on top of the [Rust programming language](https://rust-lang.org) for performance and safety. The package provides a way to store matrices in memory and perform operations on them, as well as functions for working with data frames.

`lmutils` is built primarily around the `Mat` object. These are designed to be used to perform operations on matrices without loading them into memory until necessary. This can be useful for working with lots of large matrices, like hundreds of gene blocks.

To get started with your first `Mat` object, you can use the following code:

```r
mat <- lmutils::Mat$new("matrix1.csv")
```

This will create a new `Mat` object from a file. You can then perform operations on this object, like combining it with other matrices, removing columns, or standardizing the columns. If you want this matrix to be loaded into R, you can use the `r` method:

```r
mat$combine_columns("matrix2.csv")
mat$remove_columns(c(1, 2, 3))
mat$standardize_columns()
m <- mat$r()
```

You can also pass the object directly into functions that accept a matrix convertible object, it'll then be loaded automatically (with all the stored operations applied) only when needed.

```r
lmutils::calculate_r2(
    mat,
    "outcomes1.RData",
)
```

### Example

```r
outcomes <- lmutils::Mat$new("outcomes.RData")
geneBlocks <- lapply(c(
    "geneBlock1.csv",
    "geneBlock2.csv",
    "geneBlock3.csv",
    "geneBlock4.csv",
    "geneBlock5.csv",
), function(mat) {
    mat <- lmutils::Mat$new(mat)
    mat$match_to_by_name(outcomes$col("eid"), "IID", 0)
    mat$remove_column("IID")
    mat$min_column_sum(2)
    mat$na_to_column_mean()
    mat$standardize_columns()
    mat
})
outcomes$remove_column("eid")
results <- lmutils::calculate_r2(geneBlocks, outcomes)
```

## `Mat` Objects

`lmutils::Mat` objects are a way to store matrices in memory and perform operations on them. They can be used to store operations or chain operations together for later execution. This can be useful if, for example, you wish to a hundred large matrices from files and standardize them all before using `lmutils::calculate_r2`. Using `Mat` objects, you can store the operations you wish to perform and `Mat` will execute them only when the matrix is loaded.

Passing the same `Mat` object multiple times in a single function call may cause undefined behavior. For example, the following code may not work as expected:

```r
mat <- lmutils::Mat$new("matrix1.csv")
lmutils::calculate_r2(list(mat, mat), mat)
```

### `lmutils::Mat$new`

Creates a new `Mat` object.
- `data` is a matrix convertible object.

```r
mat <- lmutils::Mat$new("matrix1.csv")
```

### `lmutils::Mat$r`

Loads the matrix from the `Mat` object.

```r
m <- mat$r()
```

### `lmutils::Mat$col`

Get a column by name or index.

```r
col <- mat$col("eid")
col <- mat$col(1)
```

### `lmutils::Mat$colnames`

Get the column names of the matrix or `NULL` if there are none.

```r
colnames <- mat$colnames()
```

### `lmutils::Mat$save`

Saves the matrix to a file.
- `file` is the file name to write to.

```r
mat$save("matrix1.mat.gz")
```

### `lmutils::Mat$combine_columns`

Combines this matrix with other matrices by columns. (`cbind`)
- `data` is a list of matrix convertible objects.

```r
mat$combine_columns("matrix2.csv")
```

### `lmutils::Mat$combine_rows`

Combines this matrix with other matrices by rows. (`rbind`)
- `data` is a list of matrix convertible objects.

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
- `other` is a matrix convertible object.
- `by` is the column index (1-based) to join by.
- `join` is the type of join to perform. 0 is inner, 1 is left, 2 is right, and 3 is full. If a row is not matched for a left or right join, it will error.

```r
mat$join("matrix2.csv", 1, 0)
```

### `lmutils::Mat$join_by_name`

Join the matrix with another matrix by a column name.
- `other` is a matrix convertible object.
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

### `lmutils::Mat$rename_column`

Rename a column by name.

```r
mat$rename_column("IID", "eid")
```

### `lmutils::Mat$rename_column_if_exists`

Rename a column by name if it exists.

```r
mat$rename_column_if_exists("IID", "eid")
```

### `lmutils::Mat$remove_duplicate_columns`

Remove columns that are duplicates of other columns. The first column is kept.

```r
mat$remove_duplicate_columns()
```

### `lmutils::Mat$remove_identical_columns`

Remove columns with all identical entries.

```r
mat$remove_identical_columns()
```

### `lmutils::Mat$eigen`

Compute the eigenvalues and eigenvectors of the matrix. The matrix must be square.

```r
eigen <- mat$eigen()
# a vector of real or complex eigenvalues
eigen$values
# a n by n matrix of real or complex eigenvectors
eigen$vectors
```

### `lmutils::Mat$subset_columns`

Subset the matrix to only include the given columns (1-based indices or names).

```r
mat$subset_columns(c(1, 2, 3))
```

### `lmutils::Mat$rename_columns_with_regex`

Rename columns with a regex and a replacement string.

```r
mat$rename_columns_with_regex("[0-9]", "X")
```

### `lmutils::Mat$scale_columns`

Scale the columns of a matrix by a given scalar or vector. The vector must be the same length as the number of columns in the matrix.

```r
mat$scale_columns(2)
mat$scale_columns(c(1, 2, 3))
```

### `lmutils::Mat$scale_rows`

Scale the rows of a matrix by a given scalar or vector. The vector must be the same length as the number of rows in the matrix.

```r
mat$scale_rows(2)
mat$scale_rows(c(1, 2, 3))
```


## Matrix Functions

### `lmutils::save`

Saves a list of matrix convertible objects to files.
- `from` is a list of matrix convertible objects.
- `to` is a character vector of file names to write to.

```r
lmutils::save(
    list("file1.csv", matrix(1:9, nrow=3), 1:3, data.frame(a=1:3, b=4:6)),
    c("file1.json", "file2.mat.gz", "file3.csv", "file4.rdata"),
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
    "mat.gz",
)
```

### `lmutils::calculate_r2`

Calculates the R^2 and adjusted R^2 values for blocks and outcomes.
- `data` is a list of matrix convertible objects.
- `outcomes` is a single matrix convertible object.
Returns a data frame with columns `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted`.

```r
results <- lmutils::calculate_r2(
    c("block1.csv", "block2.mat.gz"),
    "outcomes1.RData",
)
```

### `lmutils::column_p_values`

Compute the p value of a linear regression between each pair of columns in data and outcomes.
- `data` is a list of matrix convertible objects.
- `outcomes` is a single matrix convertible object.
The function returns a data frame with columns `p_value`, `beta`, `intercept`, `data`, `data_column`, and `outcome`.

```r
results <- lmutils::column_p_values(
    c("block1.csv", "block2.mat.gz"),
    "outcomes1.RData",
)
```

### `lmutils::linear_regression`

Perform a linear regression between each data element and each outcome column.
- `data` is a list of matrix convertible objects.
- `outcomes` is a single matrix convertible object.
The function returns a list of data frames with columns `slopes`, `intercept`, `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted` (if enabled).

```r
results <- lmutils::linear_regression(
    c("block1.csv", "block2.mat.gz"),
    "outcomes1.RData",
)
```

### `lmutils::logistic_regression`

Perform a logistic regression between each data element and each outcome column.
- `data` is a list of matrix convertible objects.
- `outcomes` is a single matrix convertible object.
The function returns a list of data frames with columns `slopes`, `intercept`, `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted` (if enabled).

```r
results <- lmutils::logistic_regression(
    c("block1.csv", "block2.mat.gz"),
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

### `lmutils::combine_rows`

Combine a potentially nested list of rows (double vectors) into a matrix.
- `data` is a list of double vectors.
- `out` is an output file name or `NULL` to return the matrix.

```r
lmutils::combine_rows(
    list(list(c(1, 2, 3)), c(4, 5, 6)),
    "combined_matrix.csv",
)
```

### `lmutils::remove_rows`

Removes rows from a matrix.
- `data` is list of matrix convertible objects.
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
- `data` is a list of matrix convertible objects.
- `out` is a standard output file.

```r
lmutils::crossprod(
    "matrix1.csv",
    "crossprod_matrix1.csv",
)
```

### `lmutils::mul`

Multiplies two matrices. Equivalent to `a %*% b`.
- `a` is a list of matrix convertible objects.
- `b` is a list of matrix convertible objects.
- `out` is a standard output file.

```r
lmutils::mul(
    "matrix1.csv",
    "matrix2.mat.gz",
    "mul_matrix1_matrix2.csv",
)
```

### `lmutils::load`

Loads a matrix convertible object into R.
- `obj` is a list matrix convertible objects.
If a single object is provided, the function will return the matrix directly, otherwise it will return a list of matrices.

```r
lmutils::load("matrix1.csv")
```

### `lmutils::match_rows`

Matches rows of a matrix by the values of a vector.
- `data` is a list of matrix convertible objects.
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
- `data` is a list of matrix convertible objects.
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

### `lmutils::df_split`

Splits a data frame into multiple data frames by a column. This function will mutably sort the data frame by the column before splitting.
- `df` is a data frame.
- `by` is the column name to split by.

```r
df <- data.frame(a=c(1, 2, 3), b=c("a", "b", "c"))
lmutils::df_split(
    df,
    "b",
)
```

### `lmutils::df_combine`

Combines a potentially nested list of data frames into a single data frame. The data frames must have the same columns.
- `data` is a list of data frames.

```r
lmutils::df_combine(
    list(data.frame(a=1:3), data.frame(a=4:6))
)
```

## Other Functions

### `lmutils::compute_r2`

Compute the R^2 value for given actual and predicted vectors.

```r
lmutils::compute_r2(
    c(1, 2, 3),
    c(1, 2, 3),
)
```

### `lmutils::compute_r2_tjur`

Compute the Tjur R^2 value for given actual and predicted vectors.

```r
lmutils::compute_r2_tjur(
    c(1, 0, 1),
    c(0.8, 0.2, 0.9),
)
```

### `lmutils::mean`

Computes the mean of a vector.

```r
lmutils::mean(
    c(1, 2, 3),
)
```

### `lmutils::median`

Computes the median of a vector.

```r
lmutils::median(
    c(1, 2, 3),
)
```

### `lmutils::sd`

Computes the standard deviation of a vector.

```r
lmutils::sd(
    c(1, 2, 3),
)
```

### `lmutils::var`

Computes the variance of a vector.

```r
lmutils::var(
    c(1, 2, 3),
)
```

### `lmutils::num_cores`

Returns the number of cores available on the system. This can be used to determine the number of cores to use for parallel operations.

```r
lmutils::num_cores()
```

## Configuration

`lmutils` exposes a number global config options that can be set using environment variables or the `lmutils` package functions:

- `LMUTILS_LOG`/`lmutils::set_log_level` to set the log level (default: `info`). Available log levels in order of increasing verbosity are `off`, `error`, `warn`, `info`, `debug`, and `trace`.
- `LMUTILS_CORE_PARALLELISM`/`lmutils::set_core_parallelism` to set the core parallelism (default: `16`). This is the number of primary operations to run in parallel.
- `LMUTILS_NUM_WORKER_THREADS`/`lmutils::set_num_worker_threads` to set the number of worker threads to use (default: `num_cpus::get() / 2`). This is the number of threads to use for parallel operations. Once an operation has been run, this value cannot be changed.
- `LMUTILS_ENABLE_PREDICTED`/`lmutils::disable_predicted`/`lmutils::enable_predicted` to enable the calculation of the predicted values in `lmutils::calculate_r2`.
- `LMUTILS_IGNORE_CORE_PARALLEL_ERRORS`/`lmutils::ignore_core_parallel_errors`/`lmutils::dont_ignore_core_parallel_errors` to ignore errors in core parallel operations. By default, if an error occurs in a core parallel operation it will be retried, if it fails its allowed number of retries then the error will be logged and the next operation will be attempted. If this option is disabled, Rust will panic after the allowed number of retries and the operation will fail.
