use core::panic;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    sync::Mutex,
};

use extendr_api::{io::Load, prelude::*, AsTypedSlice, Deref, DerefMut};
use lmutils::{File, IntoMatrix, Matrix, OwnedMatrix, ToRMatrix, Transform};
use log::{debug, error, info};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

struct RobjPar(pub Robj);

impl From<RobjPar> for Robj {
    fn from(obj: RobjPar) -> Self {
        obj.0
    }
}

impl From<Robj> for RobjPar {
    fn from(obj: Robj) -> Self {
        RobjPar(obj)
    }
}

impl Deref for RobjPar {
    type Target = Robj;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RobjPar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl Send for RobjPar {}

fn init() {
    let _ =
        env_logger::Builder::from_env(env_logger::Env::default().filter_or("LMUTILS_LOG", "info"))
            .try_init();

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("LMUTILS_NUM_WORKER_THREADS")
                .map(|s| {
                    s.parse()
                        .expect("LMUTILS_NUM_WORKER_THREADS is not a number")
                })
                .unwrap_or_else(|_| num_cpus::get() / 2)
                .clamp(1, num_cpus::get()),
        )
        .build_global();
}

fn get_num_main_threads() -> usize {
    std::env::var("LMUTILS_NUM_MAIN_THREADS")
        .map(|s| s.parse().expect("LMUTILS_NUM_MAIN_THREADS is not a number"))
        .unwrap_or(16)
        .clamp(1, num_cpus::get())
}

fn rmatrix(data: Robj) -> Result<Matrix<'static>> {
    let float = RMatrix::<f64>::try_from(data);
    match float {
        Ok(float) => Ok(float.into()),
        Err(Error::TypeMismatch(data)) => Ok(RMatrix::<i32>::try_from(data)?.into_matrix()),
        Err(e) => Err(e),
    }
}

fn file_or_matrix(data: Robj) -> Result<lmutils::Matrix<'static>> {
    init();

    if data.is_string() {
        Ok(lmutils::File::from_str(data.as_str().expect("data is a string"))?.into())
    } else if data.is_matrix() {
        Ok(RMatrix::<f64>::try_from(data)
            .expect("data is a matrix")
            .into())
    } else if data.is_integer() {
        let v = data
            .as_integer_slice()
            .expect("data is an integer vector")
            .iter()
            .map(|i| *i as f64)
            .collect::<Vec<_>>();
        Ok(Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)))
    } else if data.is_real() {
        let v = data.as_real_vector().expect("data is a real vector");
        Ok(Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)))
    } else {
        Err(MUST_BE_FILE_NAME_OR_MATRIX.into())
    }
}

fn file_or_matrix_list(data: Robj) -> Result<Vec<(String, lmutils::Matrix<'static>)>> {
    if data.is_list() {
        let data = data.as_list().expect("data is a list");
        if data.len() == 0 {
            return Err(CALCULATE_R2_DATA_MUST_BE.into());
        }
        data.into_iter()
            .enumerate()
            .map(|(i, (x, r))| {
                if r.is_matrix() {
                    Ok((
                        if x.is_empty() || x == "NA" {
                            (i + 1).to_string()
                        } else {
                            x.to_string()
                        },
                        rmatrix(r)?,
                        // RMatrix::<f64>::try_from(r).expect("i is a matrix").into(),
                    ))
                } else if r.is_string() {
                    Ok((
                        r.as_str().unwrap().to_string(),
                        lmutils::File::from_str(r.as_str().expect("i is a string"))?.into(),
                    ))
                } else if r.is_integer() {
                    let v = r
                        .as_integer_slice()
                        .expect("data is an integer vector")
                        .iter()
                        .map(|i| *i as f64)
                        .collect::<Vec<_>>();
                    Ok((
                        if x.is_empty() || x == "NA" {
                            (i + 1).to_string()
                        } else {
                            x.to_string()
                        },
                        Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)),
                    ))
                } else if r.is_real() {
                    let v = r.as_real_vector().expect("data is a real vector");
                    Ok((
                        if x.is_empty() || x == "NA" {
                            (i + 1).to_string()
                        } else {
                            x.to_string()
                        },
                        Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)),
                    ))
                } else {
                    Err(CALCULATE_R2_DATA_MUST_BE.into())
                }
            })
            .collect()
    } else if data.is_string() {
        let data = data.as_str_vector().expect("data is a string vector");
        data.into_iter()
            .map(|i| Ok((i.to_string(), lmutils::File::from_str(i)?.into())))
            .collect()
    } else if data.is_matrix() {
        let data = (
            "1".to_string(),
            RMatrix::<f64>::try_from(data)
                .expect("data is a matrix")
                .into(),
        );
        Ok(vec![data])
    } else if data.is_integer() {
        let v = data
            .as_integer_slice()
            .expect("data is an integer vector")
            .iter()
            .map(|i| *i as f64)
            .collect::<Vec<_>>();
        Ok(vec![(
            "1".to_string(),
            Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)),
        )])
    } else if data.is_real() {
        let v = data.as_real_vector().expect("data is a real vector");
        Ok(vec![(
            "1".to_string(),
            Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)),
        )])
    } else {
        Err(CALCULATE_R2_DATA_MUST_BE.into())
    }
}

fn maybe_mutating_return(
    mut data: Matrix<'static>,
    out: Nullable<Robj>,
    f: impl FnOnce(Matrix<'static>) -> Result<Matrix<'static>>,
) -> Result<Robj> {
    if let NotNull(ref out) = out {
        if out.is_string() {
            data.into_owned()?;
        } else if out.is_logical() {
            if out.as_logical().unwrap().is_true() {
                data.into_owned()?;
            }
        } else {
            return Err("out must be a string or a logical".into());
        }
    }
    let mat = f(data)?;
    if let NotNull(out) = out {
        if out.is_string() {
            File::from_str(out.as_str().unwrap())?.write_matrix(&mat.to_owned()?)?;
        } else if out.is_logical() && out.as_logical().unwrap().is_true() {
            return Ok(mat.into_robj()?);
        }
    }
    Ok(().into())
}

fn list_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    Ok(std::fs::read_dir(dir)?
        .flat_map(|entry| {
            entry.map(|entry| entry.path()).map(|path| {
                if path.is_file() {
                    Ok(vec![path])
                } else {
                    list_files(&path)
                }
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>())
}

/// Convert files from one format to another.
/// `from` and `to` must be character vectors of the same length.
/// @export
#[extendr]
pub fn convert_file(from: &[Rstr], to: &[Rstr]) -> Result<()> {
    init();

    if from.len() != to.len() {
        return Err("from and to must be the same length".into());
    }

    for (from, to) in from.iter().zip(to.iter()) {
        lmutils::convert_file(from.as_str(), to.as_str(), lmutils::TransitoryType::Float)?;
    }

    Ok(())
}

/// Convert files from one format to another.
/// `from` and `to` must be character vectors of the same length.
/// @export
#[extendr]
pub fn convert_files(from: &[Rstr], to: &[Rstr], item_type: lmutils::TransitoryType) -> Result<()> {
    init();

    if from.len() != to.len() {
        return Err("from and to must be the same length".into());
    }

    for (from, to) in from.iter().zip(to.iter()) {
        lmutils::convert_file(from.as_str(), to.as_str(), item_type)?;
    }

    Ok(())
}

/// Calculate R^2 and adjusted R^2 for a block and outcomes.
/// `data` is a character vector of file names, a list of matrices, or a single matrix.
/// `outcomes` is a file name or a matrix.
/// Returns a data frame with columns `r2`, `adj_r2`, `data`, and `outcome`.
/// @export
#[extendr]
pub fn calculate_r2(data: Robj, outcomes: Robj) -> Result<Robj> {
    init();

    debug!("Loading outcomes");
    let outcomes = file_or_matrix(outcomes)?;
    debug!("Loading data");
    let data = file_or_matrix_list(data)?;
    debug!("Loaded data");
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
        r2 = res.iter().map(|r| r.r2()).collect::<Vec<_>>(),
        adj_r2 = res.iter().map(|r| r.adj_r2()).collect::<Vec<_>>(),
        data = res.iter().map(|r| r.data()).collect::<Vec<_>>(),
        outcome = res.iter().map(|r| r.outcome()).collect::<Vec<_>>(),
        n = res.iter().map(|r| r.n()).collect::<Vec<_>>(),
        m = res.iter().map(|r| r.m()).collect::<Vec<_>>(),
        predicted = res.iter().map(|_| 0).collect::<Vec<_>>()
    )
    .as_list()
    .unwrap();
    let predicted = List::from_values(res.iter().map(|r| r.predicted())).into_robj();
    let ncols = df.len();
    df.set_elt(ncols - 1, predicted).unwrap();
    Ok(df.into_robj())
}

/// Calculate R^2 and adjusted R^2 for ranges of a data matrix and outcomes.
/// `data` is a string file name or a matrix.
/// `outcomes` is a string file name or a matrix.
/// `ranges` is a matrix with 2 columns, the start and end columns to use (inclusive).
/// Returns a data frame with columns `r2`, `adj_r2`, and `outcome` corresponding to each range.
/// @export
#[extendr]
pub fn calculate_r2_ranges(data: Robj, outcomes: Robj, ranges: RMatrix<u32>) -> Result<Robj> {
    init();

    let outcomes = file_or_matrix(outcomes)?;
    let data = file_or_matrix(data)?;
    if ranges.ncols() != 2 {
        return Err("ranges must have 2 columns".into());
    }
    if ranges.nrows() == 0 {
        return Err("ranges must have at least 1 row".into());
    }

    let data = data.transform()?;
    let data = data.as_mat_ref()?;
    let data = ranges
        .data()
        .par_chunks_exact(2)
        .map(|i| {
            let start = i[0] as usize;
            let end = i[1] as usize;
            let r = data.get(.., start..=end);
            Matrix::Ref(unsafe {
                faer::mat::from_raw_parts_mut(
                    r.as_ptr() as *mut f64,
                    r.nrows(),
                    r.ncols(),
                    r.row_stride(),
                    r.col_stride(),
                )
            })
        })
        .collect::<Vec<_>>();
    let res = lmutils::calculate_r2s(data, outcomes, None)?;

    let mut df = data_frame!(
        r2 = res.iter().map(|r| r.r2()).collect::<Vec<_>>(),
        adj_r2 = res.iter().map(|r| r.adj_r2()).collect::<Vec<_>>(),
        data = res.iter().map(|r| r.data()).collect::<Vec<_>>(),
        outcome = res.iter().map(|r| r.outcome()).collect::<Vec<_>>(),
        n = res.iter().map(|r| r.n()).collect::<Vec<_>>(),
        m = res.iter().map(|r| r.m()).collect::<Vec<_>>(),
        predicted = res.iter().map(|_| 0).collect::<Vec<_>>()
    )
    .as_list()
    .unwrap();
    let predicted = List::from_values(res.iter().map(|r| r.predicted())).into_robj();
    let ncols = df.len();
    df.set_elt(ncols - 1, predicted).unwrap();
    Ok(df.into_robj())
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
    init();

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
        data.next().expect("data has at least 1 element")
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

const MUST_BE_FILE_NAME_OR_MATRIX: &str = "must be a string file name or a matrix";
const CALCULATE_R2_DATA_MUST_BE: &str =
    "data must be a character vector, a list of matrices, or a single matrix";

/// Remove rows from a matrix.
/// `data` is a character vector of file names, a list of matrices, or a single matrix.
/// `rows` is a vector of row indices to remove (1-based).
/// `out` is a file name to write the matrix with the rows removed to.
/// If `out` is `NULL`, the matrix with the rows removed is returned otherwise `NULL`.
/// @export
#[extendr]
pub fn remove_rows(data: Robj, rows: &[u32], out: Nullable<&str>) -> Result<Nullable<Robj>> {
    init();

    let data = file_or_matrix(data)?;
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

/// Save a matrix to a file.
/// `mat` must be a double matrix.
/// `out` is the name of the file to save to.
/// @export
#[extendr]
pub fn save_matrix(mat: RMatrix<f64>, out: &str) -> Result<()> {
    init();

    Ok(File::from_str(out)?.write_matrix(&Matrix::from(mat).to_owned()?)?)
}

/// Convert a data frame to a file.
/// `df` must be a numeric data frame, numeric matrix, or a string RData file name.
/// `out` is the name of the file to save to.
/// If `out` is `NULL`, the matrix is returned otherwise `NULL`.
/// @export
#[extendr]
pub fn to_matrix(df: Robj, out: Nullable<&str>) -> Result<Nullable<RMatrix<f64>>> {
    init();

    if df.is_string() {
        let mut reader = flate2::read::GzDecoder::new(std::io::BufReader::new(
            std::fs::File::open(df.as_str().unwrap()).unwrap(),
        ));
        let mut buf = [0; 5];
        reader.read_exact(&mut buf).unwrap();
        if buf != *b"RDX3\n" {
            return Err("invalid RData file".into());
        }
        let obj = Robj::from_reader(&mut reader, extendr_api::io::PstreamFormat::XdrFormat, None)
            .unwrap();
        let obj = obj.as_pairlist().unwrap().into_iter().next().unwrap().1;
        to_matrix(obj, out)
    } else {
        let matrix = R!("as.matrix(sapply({{df}}, as.double))")?
            .as_matrix()
            .unwrap();
        if let NotNull(out) = out {
            save_matrix(matrix, out)?;
            Ok(Nullable::Null)
        } else {
            Ok(Nullable::NotNull(matrix))
        }
    }
}

/// Computes the cross product of the matrix. Equivalent to `t(data) %*% data`.
/// `data` must be a string file name or a matrix.
/// `out` is the name of the file to save to.
/// If `out` is `NULL`, the cross product is returned otherwise `NULL`.
/// @export
#[extendr]
pub fn crossprod(data: Robj, out: Nullable<&str>) -> Result<Nullable<RMatrix<f64>>> {
    init();

    let data = file_or_matrix(data)?;
    let mut mat = lmutils::cross_product(data.as_mat_ref()?);
    let mat = Matrix::Ref(mat.as_mut()).to_owned()?;
    if let NotNull(out) = out {
        File::from_str(out)?.write_matrix(&mat)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(mat.to_rmatrix()))
    }
}

/// Recursively converts a directory of RData files to matrices.
/// `from` is the directory to read from.
/// `to` is the directory to write to.
/// `file_type` is the file extension to write as.
/// If `to` is `NULL`, the files are written to `from`.
/// @export
#[extendr]
pub fn to_matrix_dir(from: &str, to: Nullable<&str>, file_type: &str) -> Result<()> {
    init();

    let to = Path::new(match to {
        Null => from,
        NotNull(to) => to,
    });
    debug!("converting files from {} to {}", from, to.display());
    std::fs::create_dir_all(to).unwrap();
    let files = Mutex::new(list_files(Path::new(from)).unwrap());
    std::thread::scope(|s| {
        for _ in 0..get_num_main_threads() {
            s.spawn(|| loop {
                let mut guard = files.lock().unwrap();
                let file = guard.pop();
                drop(guard);
                if let Some(file) = file {
                    let from_file = file.to_str().unwrap();
                    let to_file = to
                        .join(
                            from_file
                                .strip_prefix(from)
                                .unwrap()
                                .trim_matches('/')
                                .trim_matches('\\'),
                        )
                        .with_extension(file_type);
                    debug!("converting {} to {}", from_file, to_file.display());
                    std::fs::create_dir_all(to_file.parent().unwrap()).unwrap();
                    let to_file = to_file.to_str().unwrap();
                    let output = Command::new("Rscript")
                        .arg("-e")
                        .arg(format!(
                            "lmutils::to_matrix('{}', '{}')",
                            from_file, to_file
                        ))
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .output()
                        .expect("failed to execute process");
                    if output.status.code().is_none() || output.status.code().unwrap() != 0 {
                        error!("failed to convert {}", from_file);
                        error!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
                        error!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
                    } else {
                        info!("converted {} to {}", from_file, to_file)
                    }
                } else {
                    break;
                }
            });
        }
    });

    Ok(())
}

/// Standardize a matrix. All NaN values are replaced with the mean of the column and each column is scaled to have a mean of 0 and a standard deviation of 1.
/// `data` is a string file name or a matrix.
/// `out` is a file name to write the normalized matrix to, `TRUE` to return the normalized matrix
/// instead of mutating, or `NULL` to mutate the matrix passed in if it's an R matrix.
/// @export
#[extendr]
pub fn standardize(data: Robj, out: Nullable<Robj>) -> Result<Robj> {
    init();

    let data = file_or_matrix(data)?;
    maybe_mutating_return(data, out, |data| {
        Ok(data.nan_to_mean().standardization().transform()?)
    })
}

/// Load a matrix from a file.
/// `file` is the name of the file to load from.
/// @export
#[extendr]
pub fn load_matrix(file: &str) -> Result<RMatrix<f64>> {
    init();

    Ok(File::from_str(file)?.read_matrix(true)?.to_rmatrix())
}

/// Compute the p value of a linear regression between each pair of columns in two matrices.
/// `data` is a character vector of file names, a list of matrices, or a single matrix.
/// `outcomes` is a file name or a matrix.
/// Returns a data frame with columns `p`, `data`, `data_column`, and `outcome`.
/// @export
#[extendr]
pub fn column_p_values(data: Robj, outcomes: Robj) -> Result<Robj> {
    init();

    debug!("Loading outcomes");
    let outcomes = file_or_matrix(outcomes)?;
    debug!("Loading data");
    let data = file_or_matrix_list(data)?;
    debug!("Loaded data");
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

/// Match the rows of a matrix to the values in a vector by a column.
/// `data` is a string file name or a matrix.
/// `with` is a numeric vector.
/// `by` is the column to match by.
/// `out` is a file name to write the matched matrix to or `NULL` to return the matched matrix.
/// @export
#[extendr]
pub fn match_rows(data: Robj, with: &[f64], by: &str, out: Nullable<&str>) -> Result<Robj> {
    init();

    let data = file_or_matrix(data)?;
    let mut data = data.to_owned()?;
    data.match_to(with, by);
    if let NotNull(out) = out {
        File::from_str(out)?.write_matrix(&data)?;
        Ok(().into())
    } else {
        Ok(data.into_matrix().into_robj()?)
    }
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
    let with = with
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");

    debug!("matching files from {} to {}", from, to.display());
    std::fs::create_dir_all(to).unwrap();
    let files = Mutex::new(list_files(Path::new(from)).unwrap());
    std::thread::scope(|s| {
        for _ in 0..get_num_main_threads() {
            s.spawn(|| loop {
                let mut guard = files.lock().unwrap();
                let file = guard.pop();
                drop(guard);
                if let Some(file) = file {
                    let from_file = file.to_str().unwrap();
                    let to_file = to.join(
                        from_file
                            .strip_prefix(from)
                            .unwrap()
                            .trim_matches('/')
                            .trim_matches('\\'),
                    );
                    debug!("matching {} as {}", from_file, to_file.display());
                    std::fs::create_dir_all(to_file.parent().unwrap()).unwrap();
                    let to_file = to_file.to_str().unwrap();
                    let output = Command::new("Rscript")
                        .arg("-e")
                        .arg(format!(
                            "lmutils::match_rows('{}', c({}), '{}', '{}')",
                            from_file, with, by, to_file
                        ))
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .output()
                        .expect("failed to execute process");
                    if output.status.code().is_none() || output.status.code().unwrap() != 0 {
                        error!("failed to match {}", from_file);
                        error!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
                        error!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
                    } else {
                        info!("matched {} as {}", from_file, to_file)
                    }
                } else {
                    break;
                }
            });
        }
    });

    Ok(())
}

/// Compute a new column for a data frame from a regex and an existing column.
/// `df` is a data frame.
/// `column` is the column to match.
/// `regex` is the regex to match, the first capture group is used.
/// `new_column` is the new column to create.
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

/// Converts two character vectors to a list with the first vector as the names and the second as the values,
/// only keeping the first occurrence of each name. Essentially creates a hash map.
/// `names` is a character vector of names.
/// `values` is a list of values.
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

/// Add a new column to a data frame from a list of values, matching by the names of the values.
/// `df` is a data frame.
/// `column` is the column to match.
/// `values` is a list of values (a map like from `lmutils::map_from_pairs`)
/// `new_column` is the new column to create.
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

/// Add a new column to a data frame from two character vectors of names and values, matching by
/// the names.
/// `df` is a data frame.
/// `column` is the column to match.
/// `names` is a character vector of names.
/// `values` is a character vector of values.
/// `new_column` is the new column to create.
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
}

/// Mutably sorts a data frame by multiple columns in ascending order. All columns must be
/// be numeric (double or integer), character, or logical vectors.
/// `df` is a data frame.
/// `columns` is a character vector of columns to sort by.
/// @export
#[extendr]
pub fn df_sort_asc(df: List, columns: &[Rstr]) -> Result<Robj> {
    let mut names = df.iter().map(|(n, r)| (n, RobjPar(r))).collect::<Vec<_>>();
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
        }
    });
    Ok(().into())
}

/// Combine a list of double vectors into a matrix.
/// `data` is a list of double vectors.
/// `out` is a file name to write the matrix to or `NULL` to return the matrix.
/// @export
#[extendr]
pub fn combine_vectors(data: List, out: Nullable<&str>) -> Result<Nullable<RMatrix<f64>>> {
    let ncols = data.len();
    let nrows = data
        .iter()
        .map(|(_, v)| {
            v.as_real_vector()
                .expect("all vectors must be doubles")
                .len()
        })
        .next()
        .unwrap_or(0);
    let data = data.iter().map(|(_, v)| RobjPar(v)).collect::<Vec<_>>();
    let mut mat = vec![0.0; ncols * nrows];
    mat.par_chunks_exact_mut(nrows)
        .zip(data.into_par_iter())
        .for_each(|(data, v)| {
            let v = v.as_real_slice().expect("all vectors must be doubles");
            if v.len() != nrows {
                panic!("all vectors must have the same length");
            }
            data.copy_from_slice(v);
        });

    let mat = Matrix::Owned(OwnedMatrix::new(nrows, ncols, mat, None));
    if let NotNull(out) = out {
        save_matrix(mat.to_rmatrix()?, out)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(mat.to_rmatrix()?))
    }
}

/// Set the log level.
/// `level` is the log level.
/// @export
#[extendr]
pub fn set_log_level(level: &str) {
    std::env::set_var("LMUTILS_LOG", level);
}

/// Set the number of main threads to use.
/// This is the number of primary operations to perform at once.
/// `num` is the number of main threads.
/// @export
#[extendr]
pub fn set_num_main_threads(num: u32) {
    std::env::set_var("LMUTILS_NUM_MAIN_THREADS", num.to_string());
}

/// Set the number of worker threads to use.
/// This is the number of threads to use for parallel operations.
/// `num` is the number of worker threads.
/// @export
#[extendr]
pub fn set_num_worker_threads(num: u32) {
    std::env::set_var("LMUTILS_NUM_WORKER_THREADS", num.to_string());
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod lmutils;
    fn convert_file;
    fn convert_files;
    fn calculate_r2;
    fn calculate_r2_ranges;
    fn combine_matrices;
    fn remove_rows;
    fn save_matrix;
    fn to_matrix;
    fn crossprod;
    fn to_matrix_dir;
    fn standardize;
    fn load_matrix;
    fn column_p_values;
    fn match_rows;
    fn match_rows_dir;
    fn new_column_from_regex;
    fn map_from_pairs;
    fn new_column_from_map;
    fn new_column_from_map_pairs;
    fn df_sort_asc;
    fn combine_vectors;

    fn set_log_level;
    fn set_num_main_threads;
    fn set_num_worker_threads;
}
