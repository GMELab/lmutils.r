use core::panic;
use std::{collections::HashSet, str::FromStr};

use extendr_api::prelude::*;
use lmutils::Transform;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

/// Convert files from one format to another.
/// `from` and `to` must be character vectors of the same length.
/// @export
#[extendr]
pub fn convert_files(from: &[Rstr], to: &[Rstr], item_type: lmutils::TransitoryType) -> Result<()> {
    if from.len() != to.len() {
        return Err("from and to must be the same length".into());
    }

    for (from, to) in from.iter().zip(to.iter()) {
        lmutils::convert_file(from.as_str(), to.as_str(), item_type)?;
    }

    Ok(())
}

const CALCULATE_R2_DATA_MUST_BE: &str =
    "data must be a character vector, a list of matrices, or a single matrix";

/// Calculate R^2 and adjusted R^2 for a block and outcomes.
/// `data` is a character vector of file names, a list of matrices, or a single matrix.
/// `outcomes` is a file name or a matrix.
/// Returns a data frame with columns `r2` and `adj_r2` corresponding to each outcome for each
/// block in order.
/// @export
#[extendr]
pub fn calculate_r2(data: Robj, outcomes: Robj) -> Result<Robj> {
    let outcomes: Result<lmutils::Matrix> = if outcomes.is_string() {
        Ok(lmutils::File::from_str(outcomes.as_str().expect("outcomes is a string"))?.into())
    } else if outcomes.is_matrix() {
        Ok(RMatrix::<f64>::try_from(outcomes)
            .expect("outcomes is a matrix")
            .into())
    } else {
        Err("outcomes must be a string or a matrix".into())
    };
    let outcomes = outcomes?;
    let data: Result<Vec<lmutils::Matrix>> = if data.is_list() {
        let data = data.as_list().expect("data is a list");
        if data.len() == 0 {
            return Err(CALCULATE_R2_DATA_MUST_BE.into());
        }
        data.into_iter()
            .map(|(_, i)| {
                if i.is_matrix() {
                    Ok(RMatrix::<f64>::try_from(i).expect("i is a matrix").into())
                } else if i.is_string() {
                    Ok(lmutils::File::from_str(i.as_str().expect("i is a string"))?.into())
                } else {
                    Err(CALCULATE_R2_DATA_MUST_BE.into())
                }
            })
            .collect()
    } else if data.is_string() {
        let data = data.as_str_vector().expect("data is a string vector");
        data.into_iter()
            .map(|i| Ok(lmutils::File::from_str(i)?.into()))
            .collect()
    } else if data.is_matrix() {
        let data = RMatrix::<f64>::try_from(data)
            .expect("data is a matrix")
            .into();
        Ok(vec![data])
    } else {
        Err(CALCULATE_R2_DATA_MUST_BE.into())
    };
    let data = data?;

    let res = lmutils::calculate_r2s(data, outcomes)?;
    Ok(data_frame!(
        r2 = res.iter().map(|r| r.r2()).collect::<Vec<_>>(),
        adj_r2 = res.iter().map(|r| r.adj_r2()).collect::<Vec<_>>()
    )
    .into_robj())
}

const CALCULATE_R2_RANGES_DATA_MUST_BE: &str = "data must be a string file name or a matrix";

/// Calculate R^2 and adjusted R^2 for ranges of a data matrix and outcomes.
/// `data` is a string file name or a matrix.
/// `outcomes` is a string file name or a matrix.
/// `ranges` is a matrix with 2 columns, the start and end columns to use (inclusive).
/// Returns a data frame with columns `r2` and `adj_r2` corresponding to each outcome for each
/// range in order.
/// @export
#[extendr]
pub fn calculate_r2_ranges(data: Robj, outcomes: Robj, ranges: RMatrix<u32>) -> Result<Robj> {
    let outcomes: Result<lmutils::Matrix> = if outcomes.is_string() {
        Ok(lmutils::File::from_str(outcomes.as_str().expect("outcomes is a string"))?.into())
    } else if outcomes.is_matrix() {
        Ok(RMatrix::<f64>::try_from(outcomes)
            .expect("outcomes is a matrix")
            .into())
    } else {
        Err("outcomes must be a string or a matrix".into())
    };
    let outcomes = outcomes?;
    let data: Result<lmutils::Matrix> = if data.is_string() {
        Ok(lmutils::File::from_str(data.as_str().expect("data is a string"))?.into())
    } else if data.is_matrix() {
        Ok(RMatrix::<f64>::try_from(data)
            .expect("data is a matrix")
            .into())
    } else {
        Err(CALCULATE_R2_RANGES_DATA_MUST_BE.into())
    };
    let data = data?;

    if ranges.ncols() != 2 {
        return Err("ranges must have 2 columns".into());
    }
    if ranges.nrows() == 0 {
        return Err("ranges must have at least 1 row".into());
    }

    let data = data.transform()?;
    let data = data.as_mat_ref()?;
    let outcomes = outcomes.transform()?;
    let outcomes = outcomes.as_mat_ref()?;
    let res = ranges
        .data()
        .par_chunks_exact(2)
        .flat_map(|i| {
            let start = i[0] as usize;
            let end = i[1] as usize;
            let data = data.get(.., start..=end);
            lmutils::get_r2s(data, outcomes)
        })
        .collect::<Vec<_>>();

    Ok(data_frame!(
        r2 = res.iter().map(|r| r.r2()).collect::<Vec<_>>(),
        adj_r2 = res.iter().map(|r| r.adj_r2()).collect::<Vec<_>>()
    )
    .into_robj())
}

const COMBINE_MATRICES_DATA_MUST_BE: &str =
    "data must be a character vector of file names or a list of matrices";

/// Combine matrices into a single matrix.
/// `data` is a character vector of file names or a list of matrices.
/// `out` is a file name to write the combined matrix to.
/// If `out` is `NULL`, the combined matrix is returned otherwise `NULL`.
/// @export
#[extendr]
pub fn combine_matrices(data: Robj, out: Nullable<&str>) -> Result<Nullable<Robj>> {
    let out = match out {
        Null => None,
        NotNull(out) => Some(lmutils::File::from_str(out)?),
    };
    let data: Result<Vec<lmutils::Matrix>> = if data.is_list() {
        let data = data.as_list().expect("data is a list");
        if data.len() == 0 {
            return Err(COMBINE_MATRICES_DATA_MUST_BE.into());
        }
        data.into_iter()
            .map(|(_, i)| {
                if i.is_matrix() {
                    Ok(RMatrix::<f64>::try_from(i).expect("i is a matrix").into())
                } else if i.is_string() {
                    Ok(lmutils::File::from_str(i.as_str().expect("i is a string"))?.into())
                } else {
                    Err(COMBINE_MATRICES_DATA_MUST_BE.into())
                }
            })
            .collect()
    } else {
        Err(COMBINE_MATRICES_DATA_MUST_BE.into())
    };
    let data = data?;
    let res = if data.len() == 1 {
        let mut data = data.into_iter();
        let first = data.next().expect("data has at least 1 element");
        first.transform()?
    } else {
        let mut data = data.into_iter();
        let first = data.next().expect("data has at least 1 element");
        first.combine(data.collect())?
    };
    if let Some(out) = out {
        out.write_matrix(&res.to_owned()?)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(res.into_robj()?))
    }
}

const REMOVE_ROWS_DATA_MUST_BE: &str = "data must be a string file name or a matrix";

/// Remove rows from a matrix.
/// `data` is a character vector of file names, a list of matrices, or a single matrix.
/// `rows` is a vector of row indices to remove (1-based).
/// `out` is a file name to write the matrix with the rows removed to.
/// If `out` is `NULL`, the matrix with the rows removed is returned otherwise `NULL`.
/// @export
#[extendr]
pub fn remove_rows(data: Robj, rows: &[u32], out: Nullable<&str>) -> Result<Nullable<Robj>> {
    let data: Result<lmutils::Matrix> = if data.is_string() {
        Ok(lmutils::File::from_str(data.as_str().expect("data is a string"))?.into())
    } else if data.is_matrix() {
        Ok(RMatrix::<f64>::try_from(data)
            .expect("data is a matrix")
            .into())
    } else {
        Err(REMOVE_ROWS_DATA_MUST_BE.into())
    };
    let data = data?;
    let out = match out {
        Null => None,
        NotNull(out) => Some(lmutils::File::from_str(out)?),
    };
    let res = data.remove_rows(&HashSet::from_iter(rows.iter().map(|i| (i - 1) as usize)))?;
    if let Some(out) = out {
        out.write_matrix(&res.to_owned()?)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(res.into_robj()?))
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod lmutils;
    fn convert_files;
    fn calculate_r2;
    fn calculate_r2_ranges;
    fn combine_matrices;
    fn remove_rows;
}
