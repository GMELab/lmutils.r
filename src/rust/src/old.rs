/// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
/// `data` is a list of file names, a list of matrices, or a single matrix.
/// `outcomes` is a file name or a matrix.
/// Returns a data frame with columns `r2` and `adj_r2`.
#[extendr]
fn calculate_r2(data: Robj, outcomes: Robj) -> Result<Robj> {
    calculate_r2_ranges(data, outcomes, Null)
}

const DATA_MUST_BE: &str =
    "data must be a character vector, a list of matrices, or a single matrix";

/// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
/// `data` is a list of file names, a list of matrices, or a single matrix.
/// `outcomes` is a file name or a matrix.
/// `ranges` is a matrix with 2 columns, the start and end columns to use.
/// If `ranges` is not null and data is a single matrix, then the data matrix is split into ranges.
/// Returns a data frames with columns `r2` and `adj_r2`.
#[extendr(use_try_from = true)]
fn calculate_r2_ranges(data: Robj, outcomes: Robj, ranges: Nullable<RMatrix<u32>>) -> Result<Robj> {
    // data is either a list of strings, a list of matrices, a single matrix.
    // outcomes is a string or a matrix
    let outcomes: Mats = if outcomes.is_string() {
        File::from_str(outcomes.as_str().unwrap())
            .unwrap()
            .read_matrix(true)
            .into()
    } else if outcomes.is_matrix() {
        RMatrix::<f64>::try_from(outcomes).unwrap().into()
    } else {
        return Err("outcomes must be a string or a matrix".into());
    };
    let outcomes = outcomes.as_mat_ref();
    let data: Vec<Mats> = if data.is_list() {
        let data = data.as_list().unwrap();
        if data.len() == 0 {
            return Err(DATA_MUST_BE.into());
        }
        if data.iter().all(|(_, i)| i.is_matrix()) {
            data.into_iter()
                .map(|(_, i)| RMatrix::<f64>::try_from(i).unwrap().into())
                .collect()
        } else if data.iter().all(|(_, i)| i.is_string()) {
            data.iter()
                .map(|(_, i)| {
                    File::from_str(i.as_str().unwrap())
                        .unwrap()
                        .read_matrix(true)
                        .into()
                })
                .collect()
        } else {
            return Err(DATA_MUST_BE.into());
        }
    } else if data.is_string() {
        let data = data.as_str_vector().unwrap();
        data.into_iter()
            .map(|i| File::from_str(i).unwrap().read_matrix(true).into())
            .collect()
    } else if data.is_matrix() {
        let data = RMatrix::<f64>::try_from(data).unwrap();
        if let NotNull(ranges) = ranges {
            if ranges.ncols() != 2 {
                return Err("ranges must have 2 columns".into());
            }
            let data = data.as_mat_ref();
            let vec = ranges
                .data()
                .par_chunks_exact(2)
                .flat_map(|i| get_r2s(data.get(.., i[0] as usize..i[1] as usize), outcomes))
                .collect::<Vec<_>>();
            return Ok(R2::vec_to_df(vec));
        }
        vec![data.into()]
    } else {
        return Err(DATA_MUST_BE.into());
    };

    let data = data.iter().map(|i| i.as_mat_ref()).collect::<Vec<_>>();

    let vec = data
        .into_par_iter()
        .flat_map(|i| get_r2s(i, outcomes))
        .collect::<Vec<_>>();
    Ok(R2::vec_to_df(vec))
}

/// Combine a character vector of file names into a single matrix.
/// `cols` is a character vector of column names.
/// `out` is the file to write.
#[extendr]
pub fn combine_matrices(files: &[Rstr], out: &str) -> Result<()> {
    // list of genes
    // for each gene, retrieve the file name
    // read the file, it's the column of the matrix
    let out: File = out.parse()?;
    let data = files
        .iter()
        .map(|x| x.as_str())
        // .collect::<Vec<_>>()
        // .into_par_iter()
        .map(|x| {
            let Matrix { data, rows, .. } = File::from_str(x).unwrap().read_matrix::<f64, _>(true);
            (rows, data)
        })
        .collect::<Vec<_>>();
    let rows = data[0].0;
    let data = data
        .into_iter()
        .flat_map(|(_, data)| data)
        .collect::<Vec<_>>();
    out.write_matrix(&Matrix {
        data,
        rows,
        cols: files.len(),
    });
    Ok(())
}

/// Remove rows.
/// `file` is the file to read.
/// `out` is the file to write.
/// `rows` is a vector of row numbers (1-based) to remove.
#[extendr]
pub fn remove_rows(file: &str, out: &str, rows: &[u32], trans: TransitoryType) -> Result<()> {
    let rows = rows
        .iter()
        .map(|i| (i - 1) as usize)
        .collect::<HashSet<_>>();
    let data = File::from_str(file).unwrap().read_transitory(trans);
    let out = File::from_str(out).unwrap();
    let data = data.remove_rows(&rows);
    out.write_transitory(&data);
    Ok(())
}
