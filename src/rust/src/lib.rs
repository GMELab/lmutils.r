#![allow(non_snake_case)]
#![allow(deprecated)]
mod utils;

use core::panic;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    mem::MaybeUninit,
    path::Path,
    str::FromStr,
};

pub use crate::utils::{
    from_to_file, get_core_parallelism, init, list_files, matrix, maybe_mutating_return,
    maybe_return_vec, named_matrix_list, parallelize, Mat, Par,
};
use extendr_api::{prelude::*, AsTypedSlice};
use lmutils::{File, IntoMatrix, Join, Matrix, OwnedMatrix};
use rayon::{prelude::*, slice::ParallelSliceMut};
use tracing::{debug, info};
use utils::{matrix_list, maybe_return_paired};

// MATRIX OBJECT
type Ptr = ExternalPtr<utils::Mat>;

/// `lmutils::Mat` objects are a way to store matrices in memory and perform operations on them. They can be used to store operations or chain operations together for later execution. This can be useful if, for example, you wish to a hundred large matrices from files and standardize them all before using `lmutils::calculate_r2`. Using `Mat` objects, you can store the operations you wish to perform and `Mat` will execute them only when the matrix is loaded.
/// @export
#[extendr]
impl Mat {
    /// Create a new matrix object from a matrix convertable object.
    /// `data` is a matrix convertable object.
    /// @export
    pub fn new(data: Robj) -> Result<Self> {
        Ok(Self::Own(lmutils::Matrix::from_robj(data)?))
    }

    /// Load this matrix into an R matrix.
    /// @export
    pub fn r(&mut self) -> Result<RMatrix<f64>> {
        Ok(self.to_rmatrix()?)
    }

    /// Loads the matrix and gets a column by name or index.
    /// `column` is the name or index of the column to get.
    /// @export
    pub fn col(&mut self, column: Robj) -> Result<Robj> {
        let column = if column.is_integer() {
            column.as_integer().unwrap() as usize - 1
        } else if column.is_string() {
            let column = column.as_str().unwrap();
            self.column_index(column)?
        } else {
            return Err("column must be a string or integer".into());
        };
        let mat: &mut lmutils::Matrix = &mut *self;
        Ok(mat
            .col(column)?
            .map(|x| x.to_vec().into_robj())
            .unwrap_or(().into()))
    }

    /// Save this matrix to a file.
    /// `file` is the name of the file to save to.
    /// @export
    pub fn save(&mut self, file: &str) -> Result<()> {
        Ok(File::from_str(file)?.write(self)?)
    }

    /// Combine this matrix with other matrices by columns. (`cbind`)
    /// `data` is a list of matrix convertable objects.
    /// @export
    pub fn combine_columns(&mut self, data: Robj) -> Result<Ptr> {
        self.t_combine_columns(matrix_list(data)?);
        Ok(self.ptr())
    }

    /// Combine this matrix with other matrices by rows. (`rbind`)
    /// `data` is a list of matrix convertable objects.
    /// @export
    pub fn combine_rows(&mut self, data: Robj) -> Result<Ptr> {
        self.t_combine_rows(matrix_list(data)?);
        Ok(self.ptr())
    }

    /// Remove columns from this matrix.
    /// `columns` is a numeric vector of column indices to remove (1-based).
    /// @export
    pub fn remove_columns(&mut self, columns: Robj) -> Result<Ptr> {
        let columns = if columns.is_integer() {
            columns
                .as_integer_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<HashSet<_>>()
        } else if columns.is_real() {
            columns
                .as_real_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<HashSet<_>>()
        } else {
            return Err("columns must be an integer vector".into());
        };
        self.t_remove_columns(columns);
        Ok(self.ptr())
    }

    /// Remove a column from this matrix by name.
    /// `column` is the name of the column to remove.
    /// @export
    pub fn remove_column(&mut self, column: &str) -> Result<Ptr> {
        self.t_remove_column_by_name(column);
        Ok(self.ptr())
    }

    /// Remove a column from this matrix by name if it exists.
    /// `column` is the name of the column to remove.
    /// @export
    pub fn remove_column_if_exists(&mut self, column: &str) -> Result<Ptr> {
        self.t_remove_column_by_name_if_exists(column);
        Ok(self.ptr())
    }

    /// Remove rows from this matrix.
    /// `rows` is a numeric vector of row indices to remove (1-based).
    /// @export
    pub fn remove_rows(&mut self, rows: Robj) -> Result<Ptr> {
        let rows = if rows.is_integer() {
            rows.as_integer_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<HashSet<_>>()
        } else if rows.is_real() {
            rows.as_real_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<HashSet<_>>()
        } else {
            return Err("rows must be an integer vector".into());
        };
        self.t_remove_rows(rows);
        Ok(self.ptr())
    }

    /// Transpose this matrix.
    /// @export
    pub fn transpose(&mut self) -> Result<Ptr> {
        self.t_transpose();
        Ok(self.ptr())
    }

    /// Sort this matrix by the column at the given index.
    /// `by` is the index of the column to sort by (1-based).
    /// @export
    pub fn sort(&mut self, by: u32) -> Result<Ptr> {
        self.t_sort_by_column(by as usize - 1);
        Ok(self.ptr())
    }

    /// Sort this matrix by the column with the given name.
    /// `by` is the name of the column to sort by.
    /// @export
    pub fn sort_by_name(&mut self, by: &str) -> Result<Ptr> {
        self.t_sort_by_column_name(by);
        Ok(self.ptr())
    }

    /// Sort this matrix by the given order of rows.
    /// `order` is a numeric vector of row indices (1-based).
    /// @export
    pub fn sort_by_order(&mut self, order: Robj) -> Result<Ptr> {
        let order = if order.is_integer() {
            order
                .as_integer_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<Vec<_>>()
        } else if order.is_real() {
            order
                .as_real_slice()
                .unwrap()
                .iter()
                .map(|x| *x as usize - 1)
                .collect::<Vec<_>>()
        } else {
            return Err("order must be an integer vector".into());
        };
        self.t_sort_by_order(order);
        Ok(self.ptr())
    }

    /// Deduplicate this matrix by a column at the given index. The first occurrence of each value is kept.
    /// `by` is the index of the column to deduplicate by (1-based).
    /// @export
    pub fn dedup(&mut self, by: u32) -> Result<Ptr> {
        self.t_dedup_by_column(by as usize - 1);
        Ok(self.ptr())
    }

    /// Deduplicate this matrix by a column with the given name. The first occurrence of each value
    /// is kept.
    /// `by` is the name of the column to deduplicate by.
    /// @export
    pub fn dedup_by_name(&mut self, by: &str) -> Result<Ptr> {
        self.t_dedup_by_column_name(by);
        Ok(self.ptr())
    }

    /// Match the rows of this matrix to the values in a vector by a column at the given index.
    /// `with` is a numeric vector.
    /// `by` is the index of the column to match by (1-based).
    /// `join` is the type of join to perform. 0 is inner, 1 is left, and 2 is right. If a row is not matched for a left or right join, it will error.
    /// @export
    pub fn match_to(&mut self, with: Robj, by: u32, join: Join) -> Result<Ptr> {
        let with = if with.is_integer() {
            with.as_integer_slice()
                .unwrap()
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<_>>()
        } else if with.is_real() {
            with.as_real_vector().unwrap()
        } else {
            return Err("with must be an integer vector".into());
        };
        self.t_match_to(with, by as usize - 1, join);
        Ok(self.ptr())
    }

    /// Match the rows of this matrix to the values in a vector by a column with the given name.
    /// `with` is a numeric vector.
    /// `by` is the name of the column to match by.
    /// `join` is the type of join to perform. 0 is inner, 1 is left, and 2 is right. If a row is not matched for a left or right join, it will error.
    /// @export
    pub fn match_to_by_name(&mut self, with: Robj, by: &str, join: Join) -> Result<Ptr> {
        let with = if with.is_integer() {
            with.as_integer_slice()
                .unwrap()
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<_>>()
        } else if with.is_real() {
            with.as_real_vector().unwrap()
        } else {
            return Err("with must be an integer vector".into());
        };
        self.t_match_to_by_column_name(with, by, join);
        Ok(self.ptr())
    }

    /// Joins this matrix with another matrix by a column at the given index.
    /// `other` is the matrix to join with.
    /// `self_by` is the index of the column to join by in this matrix (1-based).
    /// `other_by` is the index of the column to join by in the other matrix (1-based).
    /// `join` is the type of join to perform. 0 is inner, 1 is left, and 2 is right. If a row is not matched for a left or right join, it will error.
    /// @export
    pub fn join(&mut self, other: Robj, self_by: u32, other_by: u32, join: Join) -> Result<Ptr> {
        let other = matrix(other)?;
        self.t_join(other, self_by as usize - 1, other_by as usize - 1, join);
        Ok(self.ptr())
    }

    /// Joins this matrix with another matrix by a column with the given name.
    /// `other` is the matrix to join with.
    /// `by` is the name of the column to join the matrices by.
    /// `join` is the type of join to perform. 0 is inner, 1 is left, and 2 is right. If a row is not matched for a left or right join, it will error.
    /// @export
    pub fn join_by_name(&mut self, other: Robj, by: &str, join: Join) -> Result<Ptr> {
        let other = matrix(other)?;
        self.t_join_by_column_name(other, by, join);
        Ok(self.ptr())
    }

    /// Standardizes this matrix so that each column has a mean of 0 and a standard deviation of 1.
    /// @export
    pub fn standardize_columns(&mut self) -> Result<Ptr> {
        self.t_standardize_columns();
        Ok(self.ptr())
    }

    /// Standardizes this matrix so that each row has a mean of 0 and a standard deviation of 1.
    /// @export
    pub fn standardize_rows(&mut self) -> Result<Ptr> {
        self.t_standardize_rows();
        Ok(self.ptr())
    }

    /// Remove rows from this matrix that containing any NA values.
    /// @export
    pub fn remove_na_rows(&mut self) -> Result<Ptr> {
        self.t_remove_nan_rows();
        Ok(self.ptr())
    }

    /// Remove columns from this matrix that containing any NA values.
    /// @export
    pub fn remove_na_columns(&mut self) -> Result<Ptr> {
        self.t_remove_nan_columns();
        Ok(self.ptr())
    }

    /// Replace all NA values in this matrix with the given value.
    /// `value` is the value to replace NA values with.
    /// @export
    pub fn na_to_value(&mut self, value: Robj) -> Result<Ptr> {
        let value = if value.is_integer() {
            value.as_integer().unwrap() as f64
        } else if value.is_real() {
            value.as_real().unwrap()
        } else {
            return Err("value must be a numeric scalar".into());
        };
        self.t_nan_to_value(value);
        Ok(self.ptr())
    }

    /// Replace all NA values in this matrix with the mean of the column.
    /// @export
    pub fn na_to_column_mean(&mut self) -> Result<Ptr> {
        self.t_nan_to_column_mean();
        Ok(self.ptr())
    }

    /// Replace all NA values in this matrix with the mean of the row.
    /// @export
    pub fn na_to_row_mean(&mut self) -> Result<Ptr> {
        self.t_nan_to_row_mean();
        Ok(self.ptr())
    }

    /// Remove all columns from this matrix whose sum is less than the given value.
    /// `value` is the minimum sum a column must have to be kept.
    /// @export
    pub fn min_column_sum(&mut self, value: f64) -> Result<Ptr> {
        self.t_min_column_sum(value);
        Ok(self.ptr())
    }

    /// Remove all columns from this matrix whose sum is greater than the given value.
    /// `value` is the maximum sum a column must have to be kept.
    /// @export
    pub fn max_column_sum(&mut self, value: f64) -> Result<Ptr> {
        self.t_max_column_sum(value);
        Ok(self.ptr())
    }

    /// Remove all rows from this matrix whose sum is less than the given value.
    /// `value` is the minimum sum a row must have to be kept.
    /// @export
    pub fn min_row_sum(&mut self, value: f64) -> Result<Ptr> {
        self.t_min_row_sum(value);
        Ok(self.ptr())
    }

    /// Remove all rows from this matrix whose sum is greater than the given value.
    /// `value` is the maximum sum a row must have to be kept.
    /// @export
    pub fn max_row_sum(&mut self, value: f64) -> Result<Ptr> {
        self.t_max_row_sum(value);
        Ok(self.ptr())
    }

    /// Rename a column in this matrix.
    /// `old` is the name of the column to rename.
    /// `new` is the new name of the column.
    /// @export
    pub fn rename_column(&mut self, old: &str, new: &str) -> Result<Ptr> {
        self.t_rename_column(old, new);
        Ok(self.ptr())
    }

    /// Rename a column in this matrix if it exists.
    /// `old` is the name of the column to rename.
    /// `new` is the new name of the column.
    /// @export
    pub fn rename_column_if_exists(&mut self, old: &str, new: &str) -> Result<Ptr> {
        self.t_rename_column_if_exists(old, new);
        Ok(self.ptr())
    }
}

// END MATRIX OBJECT

// MATRIX FUNCTIONS

/// Saves a list of matrix convertible objects to a character vector of file names.
/// `from` is a list of matrix convertable objects.
/// `to` is a character vector of file names to write to.
/// @export
#[extendr]
pub fn save(from: Robj, to: &[Rstr]) -> Result<()> {
    let mut from = named_matrix_list(from)?;
    let to = to.iter().map(|x| x.as_str()).collect::<Vec<_>>();
    let to = to
        .into_iter()
        .map(lmutils::File::from_str)
        .collect::<std::result::Result<Vec<lmutils::File>, lmutils::Error>>()?;

    if from.len() != to.len() {
        return Err("from and to must be the same length".into());
    }

    for ((_, from), to) in from.iter_mut().zip(to.iter()) {
        to.write(from)?;
    }

    Ok(())
}

/// Recursively converts a directory of files to the selected format.
/// `from` is the directory to read from.
/// `to` is the directory to write to or `NULL` to write to `from`.
/// `file_type` is the file extension to write as.
/// @export
#[extendr]
pub fn save_dir(from: &str, to: Nullable<&str>, file_type: &str) -> Result<()> {
    init();

    let to = Path::new(match to {
        Null => from,
        NotNull(to) => to,
    });
    debug!("converting files from {} to {}", from, to.display());
    std::fs::create_dir_all(to).unwrap();
    let files = list_files(Path::new(from)).unwrap();
    parallelize(files, move |_, file| {
        let from_file = file.to_str().unwrap();
        let to_file = from_to_file(from_file, from, to, Some(file_type));
        debug!("converting {} to {}", from_file, to_file.display());
        if let Some(dir) = to_file.parent() {
            std::fs::create_dir_all(dir).map_err(lmutils::Error::Io)?;
        }
        File::from_str(to_file.to_str().unwrap())
            .unwrap()
            .write(&mut lmutils::Matrix::from_str(from_file)?)
            .unwrap();
        info!("converted {} to {}", from_file, to_file.display());
        Ok(())
    })?;

    Ok(())
}

/// Calculate R^2 and adjusted R^2 for a block and outcomes.
/// `data` is a list of matrix convertable objects.
/// `outcomes` is a matrix convertable object.
/// Returns a data frame with columns `r2`, `adj_r2`, `data`, `outcome`, `n`, `m`, and `predicted`.
/// @export
#[extendr]
pub fn calculate_r2(data: Robj, outcomes: Robj) -> Result<Robj> {
    let outcomes = matrix(outcomes)?;
    let data = named_matrix_list(data)?;
    let (data_names, data): (Vec<_>, Vec<_>) = data.into_iter().unzip();

    debug!("Calculating R^2");
    let res = lmutils::calculate_r2s(
        data,
        outcomes,
        Some(data_names.iter().map(|x| x.as_str()).collect()),
    )?;
    debug!("Calculated R^2");
    debug!("Results {:?}", res);
    let mut df = data_frame!(
        r2 = res.iter().map(|r| r.r2()).collect_robj(),
        adj_r2 = res.iter().map(|r| r.adj_r2()).collect_robj(),
        data = res.iter().map(|r| r.data()).collect_robj(),
        outcome = res.iter().map(|r| r.outcome()).collect_robj(),
        n = res.iter().map(|r| r.n()).collect_robj(),
        m = res.iter().map(|r| r.m()).collect_robj(),
        predicted = res.iter().map(|_| 0).collect::<Vec<_>>()
    )
    .as_list()
    .unwrap();

    // due to some weird stuff with the macro, we have to set the last column manually
    let predicted = List::from_values(res.iter().map(|r| r.predicted())).into_robj();
    let ncols = df.len();
    df.set_elt(ncols - 1, predicted).unwrap();

    Ok(df.into_robj())
}

/// Compute the p value of a linear regression between each pair of columns in two matrices.
/// `data` is a list of matrix convertable objects.
/// `outcomes` is a matrix convertable object.
/// Returns a data frame with columns `p`, `data`, `data_column`, and `outcome`.
/// @export
#[extendr]
pub fn column_p_values(data: Robj, outcomes: Robj) -> Result<Robj> {
    let outcomes = matrix(outcomes)?;
    let data = named_matrix_list(data)?;
    let (data_names, data): (Vec<_>, Vec<_>) = data.into_iter().unzip();

    debug!("Calculating p values");
    let res = lmutils::column_p_values(
        data,
        outcomes,
        Some(data_names.iter().map(|x| x.as_str()).collect()),
    )?;
    debug!("Calculated p values");
    debug!("Results {:?}", res);
    Ok(data_frame!(
        p_value = res.iter().map(|r| r.p_value()).collect::<Vec<_>>(),
        data = res.iter().map(|r| r.data()).collect::<Vec<_>>(),
        data_column = res.iter().map(|r| r.data_column()).collect::<Vec<_>>(),
        outcome = res.iter().map(|r| r.outcome()).collect::<Vec<_>>()
    )
    .into_robj())
}

/// Combine a list of double vectors into a matrix.
/// `data` is a list of double vectors.
/// `out` is an output file name or `NULL` to return the matrix.
/// @export
#[extendr]
pub fn combine_vectors(data: List, out: Nullable<&str>) -> Result<Nullable<RMatrix<f64>>> {
    let ncols = data.len();
    let nrows = data
        .iter()
        .map(|(_, v)| {
            v.as_real_slice()
                .expect("all vectors must be doubles")
                .len()
        })
        .next()
        .unwrap_or(0);
    let data = data.iter().map(|(_, v)| Par(v)).collect::<Vec<_>>();
    let mut mat = vec![MaybeUninit::uninit(); ncols * nrows];
    mat.par_chunks_exact_mut(nrows)
        .zip(data.into_par_iter())
        .for_each(|(data, v)| {
            let v = v.as_real_slice().expect("all vectors must be doubles");
            if v.len() != nrows {
                panic!("all vectors must have the same length");
            }
            let v: &[MaybeUninit<f64>] = unsafe { std::mem::transmute(v) };
            data.copy_from_slice(v);
        });

    let mut mat = Matrix::Owned(OwnedMatrix::new(
        nrows,
        ncols,
        unsafe {
            std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                mat,
            )
        },
        None,
    ));
    if let NotNull(out) = out {
        File::from_str(out)?.write(&mut mat)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(mat.to_rmatrix()?))
    }
}

/// Remove rows from a matrix.
/// `data` is a list of matrix convertable objects.
/// `rows` is a vector of row indices to remove (1-based).
/// `out` is a standard output file.
/// @export
#[extendr]
pub fn remove_rows(data: Robj, rows: Robj, out: Robj) -> Result<Robj> {
    let rows = if rows.is_integer() {
        rows.as_integer_slice()
            .unwrap()
            .iter()
            .map(|x| *x as usize - 1)
            .collect::<HashSet<_>>()
    } else if rows.is_real() {
        rows.as_real_slice()
            .unwrap()
            .iter()
            .map(|x| *x as usize - 1)
            .collect::<HashSet<_>>()
    } else {
        return Err("rows must be an integer vector".into());
    };
    maybe_return_paired(data, out, |mut data| {
        data.remove_rows(&rows)?;
        Ok(data)
    })
}

/// Computes the cross product of the matrix. Equivalent to `t(data) %*% data`.
/// `data` is a list of matrix convertable objects.
/// `out` is a standard output file.
/// @export
#[extendr]
pub fn crossprod(data: Robj, out: Nullable<Robj>) -> Result<Robj> {
    maybe_mutating_return(data, out, |mut data| {
        let m = data.as_mat_ref()?;
        let m = m.transpose() * m;
        Ok(m.into_matrix())
    })
}

/// Multiply two matrices. Equivalent to `a %*% b`.
/// `a` is a list of matrix convertable objects.
/// `b` is a list of matrix convertable objects.
/// `out` is a standard output file.
/// @export
#[extendr]
pub fn mul(a: Robj, b: Robj, out: Nullable<Robj>) -> Result<Robj> {
    maybe_mutating_return(a, out, |mut a| {
        let mut b = matrix(b)?;
        let m = a.as_mat_ref()? * b.as_mat_ref()?;
        Ok(m.into_matrix())
    })
}

/// Load a matrix convertable object into R.
/// `obj` is a list of matrix convertable objects.
/// If a single object is provided, the function will return the matrix directly, otherwise it will return a list of matrices.
/// @export
#[extendr]
pub fn load(obj: Robj) -> Result<Robj> {
    maybe_return_paired(obj, ().into(), Ok)
}

/// Match the rows of a matrix to the values in a vector by a column.
/// `data` is a list of matrix convertable objects.
/// `with` is a numeric vector.
/// `by` is the column to match by.
/// `out` is a standard output file.
/// @export
#[extendr]
pub fn match_rows(data: Robj, with: Robj, by: &str, out: Robj) -> Result<Robj> {
    let with = if with.is_integer() {
        with.as_integer_slice()
            .unwrap()
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<_>>()
    } else if with.is_real() {
        with.as_real_vector().unwrap()
    } else {
        return Err("with must be an integer vector".into());
    };
    maybe_return_paired(data, out, |mut data| {
        data.match_to_by_column_name(with.as_slice(), by, lmutils::Join::Inner)?;
        Ok(data)
    })
}

/// Recursively matches the rows of a directory of matrices to the values in a vector by a column.
/// `from` is the directory to read from.
/// `to` is the directory to write to.
/// `with` is a numeric vector.
/// `by` is the column to match by.
/// @export
#[extendr]
pub fn match_rows_dir(from: &str, to: &str, with: &[f64], by: &str) -> Result<()> {
    init();

    let to = Path::new(to);
    debug!("matching files from {} to {}", from, to.display());
    std::fs::create_dir_all(to).unwrap();
    let files = list_files(Path::new(from)).unwrap();
    parallelize(files, move |_, file| {
        let from_file = file.to_str().unwrap();
        let to_file = from_to_file(from_file, from, to, None);
        debug!("matching {} as {}", from_file, to_file.display());
        std::fs::create_dir_all(to_file.parent().unwrap()).unwrap();
        let to_file = to_file.to_str().unwrap();
        File::from_str(to_file)
            .unwrap()
            .write(
                lmutils::Matrix::from_str(from_file)?.match_to_by_column_name(
                    with,
                    by,
                    lmutils::Join::Inner,
                )?,
            )
            .unwrap();
        Ok(())
    })?;

    Ok(())
}

/// Deduplicate a matrix by a column. The first occurrence of each value is kept.
/// `data` is a list of matrix convertable objects.
/// `by` is the column to deduplicate by.
/// `out` is a standard output file.
/// @export
#[extendr]
pub fn dedup(data: Robj, by: &str, out: Robj) -> Result<Robj> {
    maybe_return_paired(data, out, |mut data| {
        data.dedup_by_column_name(by)?;
        Ok(data)
    })
}

// END MATRIX FUNCTIONS

// DATA FRAME FUNCTIONS

/// Compute a new column for a data frame from a regex and an existing column.
/// `df` is a data frame.
/// `column` is the column to match.
/// `regex` is the regex to match. The first capture group is used.
/// `new_column` is the new column name.
/// This function uses the Rust flavor of regex, see https://docs.rs/regex/latest/regex/#syntax for more /* information */.
/// @export
#[extendr]
pub fn new_column_from_regex(
    df: List,
    column: &str,
    regex: &str,
    new_column: &str,
) -> Result<Robj> {
    init();

    let (_, column) = df
        .iter()
        .find(|(n, _)| *n == column)
        .unwrap_or_else(|| panic!("column {} not found", column));
    let column = column
        .as_str_vector()
        .expect("column should be a character vector");
    let re = regex::Regex::new(regex).unwrap();
    let column = column
        .par_iter()
        .map(|s| {
            re.captures(s)
                .map(|c| c.get(1).expect("expected at least one match").as_str())
                .unwrap_or_else(|| panic!("could not match regex to value {}", s))
        })
        .collect::<Vec<_>>();
    let mut pairs = df.iter().collect::<Vec<_>>();
    pairs.push((new_column, column.into()));
    let names = pairs.iter().map(|(n, _)| n).collect::<Vec<_>>();
    let values = pairs.iter().map(|(_, v)| v).collect::<Vec<_>>();
    let l = List::from_names_and_values(names, values)?;
    R!("as.data.frame({{l}})")
}

/// Converts two character vectors into a named list, where the first vector is the names and the second vector is the values. Only the first occurrence of each name is used, essentially creating a map.
/// `names` is a character vector of names.
/// `values` is a character vector of values.
/// @export
#[extendr]
pub fn map_from_pairs(names: &[Rstr], values: &[Rstr]) -> Result<Robj> {
    let mut map = HashMap::new();
    for (name, value) in names.iter().zip(values.iter()).rev() {
        map.insert(name.as_str(), value.as_str());
    }
    let mut list = List::from_values(map.values());
    list.set_names(map.into_keys())?;
    Ok(list.into_robj())
}

fn new_column_from_map_aux(
    df: List,
    map: HashMap<&str, &str>,
    column: Vec<&str>,
    new_column: &str,
) -> Result<Robj> {
    let column = column
        .par_iter()
        .map(|s| {
            map.get(s)
                .unwrap_or_else(|| panic!("could not find value for {}", s))
        })
        .collect::<Vec<_>>();
    let mut pairs = df.iter().collect::<Vec<_>>();
    pairs.push((new_column, column.into()));
    let names = pairs.iter().map(|(n, _)| n).collect::<Vec<_>>();
    let values = pairs.iter().map(|(_, v)| v).collect::<Vec<_>>();
    let l = List::from_names_and_values(names, values)?;
    R!("as.data.frame({{l}})")
}

/// Compute a new column for a data frame from a list of values and an existing column, matching by the names of the values.
/// `column` is the column to match.
/// `values` is a named list of values.
/// `new_column` is the new column name.
/// @export
#[extendr]
pub fn new_column_from_map(df: List, column: &str, values: List, new_column: &str) -> Result<Robj> {
    let (_, column) = df
        .iter()
        .find(|(n, _)| *n == column)
        .unwrap_or_else(|| panic!("column {} not found", column));
    let column = column
        .as_str_vector()
        .expect("column should be a character vector");
    let mut map = HashMap::new();
    for (name, value) in values.iter() {
        map.insert(name, value.as_str().unwrap());
    }
    new_column_from_map_aux(df, map, column, new_column)
}

/// Compute a new column for a data frame from two character vectors of names and values, matching by the names.
/// `df` is a data frame.
/// `column` is the column to match.
/// `names` is a character vector of names.
/// `values` is a character vector of values.
/// `new_column` is the new column name.
/// @export
#[extendr]
pub fn new_column_from_map_pairs(
    df: List,
    column: &str,
    names: &[Rstr],
    values: &[Rstr],
    new_column: &str,
) -> Result<Robj> {
    let (_, column) = df
        .iter()
        .find(|(n, _)| *n == column)
        .unwrap_or_else(|| panic!("column {} not found", column));
    let column = column
        .as_str_vector()
        .expect("column should be a character vector");
    let mut map = HashMap::new();
    for (name, value) in names.iter().zip(values.iter()).rev() {
        map.insert(name.as_str(), value.as_str());
    }
    new_column_from_map_aux(df, map, column, new_column)
}

enum Col<'a> {
    Str(Vec<&'a str>),
    Int(&'a [i32]),
    Real(&'a [f64]),
    Logical(&'a [Rbool]),
}

unsafe impl<'a> Send for Col<'a> {}

impl<'a> Col<'a> {
    fn cmp(&self, i: usize, j: usize) -> Ordering {
        match self {
            Col::Str(v) => v[i].cmp(v[j]),
            Col::Int(v) => v[i].cmp(&v[j]),
            Col::Real(v) => v[i].partial_cmp(&v[j]).unwrap(),
            Col::Logical(v) => v[i].inner().cmp(&v[j].inner()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Col::Str(v) => v.len(),
            Col::Int(v) => v.len(),
            Col::Real(v) => v.len(),
            Col::Logical(v) => v.len(),
        }
    }

    fn cmps(&self) -> Vec<Cmp> {
        match self {
            Col::Str(v) => v.iter().map(|s| Cmp::Str(s.to_string())).collect(),
            Col::Int(v) => v.iter().map(|i| Cmp::Int(*i)).collect(),
            Col::Real(v) => v.iter().map(|f| Cmp::Real(*f)).collect(),
            Col::Logical(v) => v.iter().map(|b| Cmp::Int(b.inner() as i32)).collect(),
        }
    }
}

/// Mutably sorts a data frame in ascending order by multiple columns in ascending order.
/// `df` is a data frame.
/// `columns` is a character vector of columns to sort by. The sort columns must be numeric
/// (integer or double), character, or logical vectors.
/// @export
#[extendr]
pub fn df_sort_asc(df: List, columns: &[Rstr]) -> Result<Robj> {
    let mut names = df.iter().map(|(n, r)| (n, Par(r))).collect::<Vec<_>>();
    let cols = columns
        .iter()
        .map(|c| {
            let (_, col) = names
                .iter()
                .find(|(n, _)| *n == c.as_str())
                .unwrap_or_else(|| panic!("column {} not found", c.as_str()));
            if col.is_string() {
                Col::Str(col.as_str_vector().unwrap())
            } else if col.is_integer() {
                Col::Int(col.as_integer_slice().unwrap())
            } else if col.is_real() {
                Col::Real(col.as_real_slice().unwrap())
            } else if col.is_logical() {
                Col::Logical(col.as_logical_slice().unwrap())
            } else {
                panic!(
                    "column {} must be a character, integer, real, or logical vector",
                    c.as_str()
                )
            }
        })
        .collect::<Vec<_>>();
    let mut indices = (0..cols[0].len()).collect::<Vec<_>>();
    indices.as_parallel_slice_mut().par_sort_by(|&i, &j| {
        cols.iter().fold(Ordering::Equal, |acc, col| {
            if acc == Ordering::Equal {
                col.cmp(i, j)
            } else {
                acc
            }
        })
    });
    names.par_iter_mut().for_each(|(_, col)| {
        if col.is_string() {
            let slice: &mut [Rstr] = col.as_typed_slice_mut().unwrap();
            let new = indices
                .iter()
                .map(|&i| unsafe { slice.get_unchecked(i) }.clone())
                .collect::<Vec<_>>();
            slice.clone_from_slice(new.as_slice());
        } else if col.is_integer() {
            let slice: &mut [i32] = col.as_typed_slice_mut().unwrap();
            let new = indices.iter().map(|&i| slice[i]).collect::<Vec<_>>();
            slice.copy_from_slice(&new);
        } else if col.is_real() {
            let slice: &mut [f64] = col.as_typed_slice_mut().unwrap();
            let new = indices.iter().map(|&i| slice[i]).collect::<Vec<_>>();
            slice.copy_from_slice(&new);
        } else if col.is_logical() {
            let slice: &mut [Rbool] = col.as_typed_slice_mut().unwrap();
            let new = indices.iter().map(|&i| slice[i]).collect::<Vec<_>>();
            slice.clone_from_slice(&new);
        } else if col.is_list() {
            let mut list = col.as_list().unwrap();
            let new = indices
                .iter()
                .map(|&i| list.elt(i).unwrap().clone())
                .collect::<Vec<_>>();
            for (i, v) in new.into_iter().enumerate() {
                list.set_elt(i, v).unwrap();
            }
        } else {
            panic!("column must be a vector");
        }
    });
    Ok(().into())
}

#[derive(Clone, Debug)]
enum Cmp {
    Str(String),
    Int(i32),
    Real(f64),
}

impl Cmp {
    fn cmp(&self, other: &Cmp) -> Ordering {
        match (self, other) {
            (Cmp::Str(a), Cmp::Str(b)) => a.cmp(b),
            (Cmp::Int(a), Cmp::Int(b)) => a.cmp(b),
            (Cmp::Real(a), Cmp::Real(b)) => a.partial_cmp(b).unwrap(),
            _ => panic!("cannot compare different types"),
        }
    }

    fn into_string(self) -> String {
        match self {
            Cmp::Str(s) => s,
            Cmp::Int(i) => i.to_string(),
            Cmp::Real(f) => f.to_string(),
        }
    }
}

impl PartialEq for Cmp {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

/// Splits a data frame into multiple data frames by a column. This function will mutably sort the
/// data frame by the column before splitting.
/// `df` is a data frame.
/// `by` is the column to split by. The column must be a numeric (integer or double) or character
/// vector.
/// @export
#[extendr]
pub fn df_split(df: List, by: &str) -> Result<Robj> {
    df_sort_asc(df.clone(), &[by.into()])?;
    let (_, col) = df
        .iter()
        .find(|(n, _)| *n == by)
        .unwrap_or_else(|| panic!("column {} not found", by));
    let unique = if col.is_string() {
        let mut col = col.as_str_vector().unwrap().to_vec();
        col.dedup();
        col.len()
    } else if col.is_integer() {
        let mut col = col.as_integer_slice().unwrap().to_vec();
        col.dedup();
        col.len()
    } else if col.is_real() {
        let mut col = col.as_real_slice().unwrap().to_vec();
        col.dedup();
        col.len()
    } else {
        panic!("column must be a character, integer, or real vector");
    };
    if unique == 1 {
        return Ok(df.into_robj());
    }
    let col = if col.is_string() {
        Col::Str(col.as_str_vector().unwrap())
    } else if col.is_integer() {
        Col::Int(col.as_integer_slice().unwrap())
    } else if col.is_real() {
        Col::Real(col.as_real_slice().unwrap())
    } else {
        panic!("column must be a character, integer, or real vector");
    };
    let col = col.cmps();
    if col.is_empty() {
        return Ok(df.into_robj());
    }
    let mut dfs = Vec::with_capacity(unique);
    let mut next = 0;
    let mut start = 0;
    let mut current = col[0].clone();
    let names = df.iter().map(|(n, _)| n).collect::<Vec<_>>();
    let values = df.iter().map(|(_, v)| v).collect::<Vec<_>>();
    let mut group = |i, c: Cmp, force| -> Result<()> {
        if c != current || force {
            let subset = values
                .iter()
                .map(|v| {
                    if v.is_string() {
                        let v = v.as_str_vector().unwrap();
                        v[start..i]
                            .iter()
                            // .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .into_robj()
                    } else if v.is_integer() {
                        let v = v.as_integer_slice().unwrap();
                        v[start..i].to_vec().into_robj()
                    } else if v.is_real() {
                        let v = v.as_real_slice().unwrap();
                        v[start..i].to_vec().into_robj()
                    } else if v.is_logical() {
                        let v = v.as_logical_slice().unwrap();
                        v[start..i].to_vec().into_robj()
                    } else if v.is_list() {
                        List::from_values(
                            v.as_list()
                                .unwrap()
                                .iter()
                                .skip(start)
                                .take(i - start)
                                .map(|(_, v)| v),
                        )
                        .into_robj()
                    } else {
                        panic!("column must be a vector");
                    }
                })
                .collect::<Vec<_>>();
            let l = i - start;
            let df = List::from_names_and_values(names.clone(), subset)?;
            df.set_class(&["data.frame"])?;
            df.set_attrib(
                row_names_symbol(),
                (1..=l).map(|x| x.to_string()).collect_robj(),
            )?;
            next += 1;
            let r = std::mem::replace(&mut current, c);
            dfs.push((r, df));
            start = i;
        }
        Ok(())
    };
    let len = col.len();
    let last = col[col.len() - 1].clone();
    for (i, c) in col.into_iter().enumerate() {
        group(i, c, false)?;
    }
    group(len, last, true)?;

    Ok(List::from_pairs(
        dfs.into_iter()
            .map(|(n, df)| (n.into_string(), df.into_robj())),
    )
    .into_robj())
}

// END DATA FRAME FUNCTIONS

// CONFIG FUNCTIONS

/// Set the log level.
/// `level` is the log level.
/// @export
#[extendr]
pub fn set_log_level(level: &str) {
    std::env::set_var("LMUTILS_LOG", level);
}

/// Sets the core parallelism for lmutils.
/// This is the number of primary operations to perform at once.
/// `num` is the number of main threads.
/// @export
#[extendr]
pub fn set_core_parallelism(num: u32) {
    std::env::set_var("LMUTILS_CORE_PARALLELISM", num.to_string());
}

/// Set the number of worker threads to use. By default this value is `num_cpus::get() / 2`.
/// This is the number of threads to use for parallel operations.
/// `num` is the number of worker threads.
/// @export
#[extendr]
pub fn set_num_worker_threads(num: u32) {
    std::env::set_var("LMUTILS_NUM_WORKER_THREADS", num.to_string());
}

/// Disable the calculation of predicted values in `calculate_r2`.
/// @export
#[extendr]
pub fn disable_predicted() {
    std::env::set_var("LMUTILS_DISABLE_PREDICTED", "1");
}

/// Enable the calculation of predicted values in `calculate_r2`.
/// @export
#[extendr]
pub fn enable_predicted() {
    std::env::remove_var("LMUTILS_DISABLE_PREDICTED");
}

// END CONFIG FUNCTIONS

// INTERNAL FUNCTIONS

/// @export
#[extendr]
#[allow(unreachable_code)]
pub fn internal_lmutils_fd_into_file(file: &str, fd: i32) {
    init();

    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;

        std::env::set_var("LMUTILS_FD", "1");
        // read from the fd in uncompressed rkyv and write to the file
        let file = lmutils::File::from_str(file).unwrap();
        // std::thread::sleep(std::time::Duration::from_secs(60));
        let fd = unsafe { std::fs::File::from_raw_fd(fd) };
        let mut matrix = lmutils::File::new("", lmutils::FileType::Rkyv, false)
            .read_from_reader(fd)
            .unwrap();
        file.write(&mut matrix).unwrap();
        return;
    }
    panic!("unsupported platform")
}

/// @export
#[extendr]
#[allow(unreachable_code)]
pub fn internal_lmutils_file_into_fd(file: &str, fd: i32) {
    init();

    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;

        std::env::set_var("LMUTILS_FD", "1");
        // read from the file and write to the fd in rkyv format
        let file = lmutils::File::from_str(file).unwrap();
        let fd = unsafe { std::fs::File::from_raw_fd(fd) };
        let mut matrix = file.read().unwrap();
        lmutils::File::new("", lmutils::FileType::Rkyv, false)
            .write_matrix_to_writer(fd, &mut matrix)
            .unwrap();
        return;
    }
    panic!("unsupported platform")
}

// END INTERNAL FUNCTIONS

// DEPRECATED FUNCTIONS

/// DEPRECATED
/// Convert files from one format to another.
/// `from` is a list of matrix convertable objects.
/// `to` is a list of file names to write to.
/// @export
#[extendr]
#[deprecated]
pub fn convert_file(from: Robj, to: &[Rstr]) -> Result<()> {
    init();

    save(from, to)
}

/// DEPRECATED
/// Save a matrix to a file.
/// `mat` must be a double matrix.
/// `out` is the name of the file to save to.
/// @export
#[extendr]
#[deprecated]
pub fn save_matrix(mat: Robj, out: &[Rstr]) -> Result<()> {
    init();

    save(mat, out)
}

/// DEPRECATED
/// Convert a data frame to a file.
/// `df` must be a numeric data frame, numeric matrix, or a string RData file name.
/// `out` is the name of the file to save to.
/// If `out` is `NULL`, the matrix is returned otherwise `NULL`.
/// @export
#[extendr]
#[deprecated]
pub fn to_matrix(df: Robj, out: Nullable<Robj>) -> Result<Robj> {
    maybe_mutating_return(df, out, Ok)
}

/// Standardize a matrix. All NaN values are replaced with the mean of the column and each column is scaled to have a mean of 0 and a standard deviation of 1.
/// `data` is a string file name or a matrix.
/// `out` is a file name to write the normalized matrix to or `NULL` to return the normalized
/// matrix.
/// @export
#[extendr]
#[deprecated]
pub fn standardize(data: Robj, out: Robj) -> Result<Robj> {
    maybe_return_paired(data, out, |mut data| {
        data.standardize_columns()?;
        Ok(data)
    })
}

/// DEPRECATED
/// Recursively converts a directory of files to the selected format.
/// `from` is the directory to read from.
/// `to` is the directory to write to.
/// `file_type` is the file extension to write as.
/// If `to` is `NULL`, the files are written to `from`.
/// @export
#[extendr]
#[deprecated]
pub fn to_matrix_dir(from: &str, to: Nullable<&str>, file_type: &str) -> Result<()> {
    save_dir(from, to, file_type)
}

/// DEPRECATED
/// Extend matrices into a single matrix by rows.
/// `data` is a character vector of file names or a list of matrices.
/// `out` is a file name to write the combined matrix to.
/// If `out` is `NULL`, the combined matrix is returned otherwise `NULL`.
/// @export
#[extendr]
#[deprecated]
pub fn extend_matrices(data: Robj, out: Nullable<Robj>) -> Result<Nullable<Robj>> {
    maybe_return_vec(data, out, |mut first, mut data| {
        first.combine_rows(data.as_mut_slice())?;
        Ok(first)
    })
}

/// DEPRECATED
/// Set the number of main threads to use.
/// This is the number of primary operations to perform at once.
/// `num` is the number of main threads.
/// @export
#[extendr]
#[deprecated]
pub fn set_num_main_threads(num: u32) {
    std::env::set_var("LMUTILS_NUM_MAIN_THREADS", num.to_string());
}

/// DEPRECATED
/// Combine matrices into a single matrix by columns.
/// `data` is a character vector of file names or a list of matrices.
/// `out` is a file name to write the combined matrix to.
/// If `out` is `NULL`, the combined matrix is returned otherwise `NULL`.
/// @export
#[extendr]
#[deprecated]
pub fn combine_matrices(data: Robj, out: Nullable<Robj>) -> Result<Nullable<Robj>> {
    maybe_return_vec(data, out, |mut first, mut data| {
        first.combine_columns(data.as_mut_slice())?;
        Ok(first)
    })
}

/// DEPRECATED
/// Load a matrix from a file.
/// `file` is the name of the file to load from.
/// @export
#[extendr]
pub fn load_matrix(file: Robj) -> Result<Robj> {
    maybe_return_paired(file, ().into(), Ok)
}

// END DEPRECATED FUNCTIONS

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod lmutils;

    impl Mat;

    fn save;
    fn save_dir;
    fn calculate_r2;
    fn column_p_values;
    fn combine_vectors;
    fn remove_rows;
    fn crossprod;
    fn mul;
    fn load;
    fn match_rows;
    fn match_rows_dir;
    fn dedup;

    fn new_column_from_regex;
    fn map_from_pairs;
    fn new_column_from_map;
    fn new_column_from_map_pairs;
    fn df_sort_asc;
    fn df_split;

    fn set_log_level;
    fn set_core_parallelism;
    fn set_num_worker_threads;
    fn disable_predicted;
    fn enable_predicted;

    fn internal_lmutils_fd_into_file;
    fn internal_lmutils_file_into_fd;

    fn convert_file;
    fn save_matrix;
    fn standardize;
    fn to_matrix;
    fn to_matrix_dir;
    fn extend_matrices;
    fn set_num_main_threads;
    fn combine_matrices;
    fn load_matrix;
}
