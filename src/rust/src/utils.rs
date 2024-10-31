use extendr_api::prelude::*;
use lmutils::{File, IntoMatrix, OwnedMatrix};
use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    str::FromStr,
};

pub struct Par<T>(pub T);

impl<T> Par<T> {
    pub fn inner(self) -> T {
        self.0
    }
}

impl<T> From<T> for Par<T> {
    fn from(v: T) -> Self {
        Par(v)
    }
}

impl<T> Deref for Par<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Par<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T> Send for Par<T> {}

static INIT: std::sync::Once = std::sync::Once::new();

pub fn parallelize<T, R>(
    data: Vec<T>,
    f: impl Fn(usize, T) -> Result<R> + Send + Sync,
) -> Result<Vec<R>> {
    let mut results = Vec::with_capacity(data.len());
    results.extend((0..data.len()).map(|_| MaybeUninit::uninit()));
    let results = std::sync::Mutex::new(results);
    let data = std::sync::Mutex::new(data.into_iter().map(Par).enumerate().collect::<Vec<_>>());
    std::thread::scope(|s| {
        for _ in 0..get_core_parallelism() {
            s.spawn(|| loop {
                let (i, data) = {
                    let mut data = data.lock().unwrap();
                    if data.is_empty() {
                        break;
                    }
                    data.pop().unwrap()
                };
                let result = f(i, data.inner());
                results.lock().unwrap()[i].write(Par(result));
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|x| unsafe { x.assume_init() }.inner())
        .collect()
}

pub fn init() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .with_env_var("LMUTILS_LOG")
                    .from_env_lossy(),
            )
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
    });
}

pub fn get_core_parallelism() -> usize {
    std::env::var("LMUTILS_CORE_PARALLELISM")
        .unwrap_or_else(|_| {
            std::env::var("LMUTILS_NUM_MAIN_THREADS").unwrap_or_else(|_| "16".to_string())
        })
        .parse::<usize>()
        .expect("LMUTILS_CORE_PARALLELISM is not a number")
        .clamp(1, num_cpus::get())
}

pub fn matrix(robj: Robj) -> Result<lmutils::Matrix> {
    init();

    if robj.is_external_pointer() {
        return Ok(
            Mat::Ref(extendr_api::externalptr::ExternalPtr::<Mat>::try_from(
                robj,
            )?)
            .into_matrix(),
        );
    }

    Ok(lmutils::Matrix::from_robj(robj)?)
}

pub fn named_matrix_list(robj: Robj) -> Result<Vec<(String, lmutils::Matrix)>> {
    init();

    Ok(
        if robj.is_list()
            && !robj
                .as_list()
                .unwrap()
                .class()
                .map(|x| x.into_iter().any(|c| c == "data.frame"))
                .unwrap_or(false)
        {
            // if it's a list, and not a data frame, then it's a list of matrices
            let mut data = Vec::new();
            let mut i = 1;
            for (name, r) in robj.as_list().unwrap().into_iter() {
                let r = named_matrix_list(r)?
                    .into_iter()
                    .enumerate()
                    .map(|(j, (x, r))| {
                        if x.parse::<usize>().is_ok() {
                            ((i + j).to_string(), r)
                        } else {
                            (x, r)
                        }
                    })
                    .collect::<Vec<_>>();
                i += r.len();
                if r.len() == 1 {
                    data.push((name.to_string(), r.into_iter().next().unwrap().1));
                } else {
                    data.extend(r);
                }
            }
            data
        } else if robj.is_string() {
            // if it's a string, then it's a list of files
            robj.as_str_iter()
                .unwrap()
                .map(|x| Ok((x.to_string(), lmutils::File::from_str(x)?.into())))
                .collect::<Result<Vec<_>>>()?
        } else {
            // otherwise, it's a single matrix
            let r = matrix(robj)?;
            vec![("1".to_string(), r)]
        },
    )
}

pub fn matrix_list(robj: Robj) -> Result<Vec<lmutils::Matrix>> {
    Ok(named_matrix_list(robj)?
        .into_iter()
        .map(|(_, r)| r)
        .collect())
}

pub fn maybe_mutating_return(
    data: Robj,
    out: Nullable<Robj>,
    f: impl FnOnce(lmutils::Matrix) -> Result<lmutils::Matrix>,
) -> Result<Robj> {
    maybe_mutating_return_matrix(matrix(data)?, out, f)
}

pub fn maybe_mutating_return_matrix(
    mut data: lmutils::Matrix,
    out: Nullable<Robj>,
    f: impl FnOnce(lmutils::Matrix) -> Result<lmutils::Matrix>,
) -> Result<Robj> {
    // if out is NULL or a string, convert into_owned
    // if out is TRUE, then we mutate
    if let NotNull(out) = out {
        if out.is_string() {
            // out is a string, so we convert to owned and write to file
            let out = out.as_str().unwrap();
            let file = File::from_str(out)?;
            data.into_owned()?;
            let mut data = f(data)?;
            file.write(&mut data)?;
            Ok(().into())
        } else if out.is_logical() && out.as_logical().unwrap().is_true() {
            // out is TRUE, so we mutate if possible
            f(data)?;
            Ok(().into())
        } else {
            Err("out must be a string, TRUE, or NULL".into())
        }
    } else {
        // out is NULL, so we convert to owned and return
        data.into_owned()?;
        Ok(f(data)?.into_robj()?)
    }
}

pub fn maybe_return_vec(
    data: Robj,
    out: Nullable<Robj>,
    f: impl FnOnce(lmutils::Matrix, Vec<lmutils::Matrix>) -> Result<lmutils::Matrix>,
) -> Result<Nullable<Robj>> {
    let mut data = matrix_list(data)?;
    let out = match out {
        NotNull(out) if out.is_string() => Some(lmutils::File::from_str(out.as_str().unwrap())?),
        _ => None,
    };
    let first = data.remove(0);
    let mut result = f(first, data)?;
    if let Some(out) = out {
        out.write(&mut result)?;
        Ok(Nullable::Null)
    } else {
        Ok(Nullable::NotNull(result.into_robj()?))
    }
}

pub fn maybe_return_paired(
    data: Robj,
    out: Robj,
    f: impl Fn(lmutils::Matrix) -> Result<lmutils::Matrix> + Send + Sync,
) -> Result<Robj> {
    let data = matrix_list(data)?;
    let is_list = out.is_list();
    // out is NULL, a character vector, a logical vector, or a list
    let v = if out.is_null() {
        (0..data.len()).map(|_| ().into()).collect()
    } else if out.is_string() {
        // out is a character vector
        let out = out
            .as_str_iter()
            .unwrap()
            .map(|x| x.into_robj())
            .collect::<Vec<_>>();
        if out.len() != data.len() {
            return Err("out must have the same length as data".into());
        }
        out
    } else if out.is_logical() {
        let out = out.as_logical_iter().unwrap().collect::<Vec<_>>();
        if out.len() == 1 && out[0].is_true() {
            (0..data.len()).map(|_| true.into()).collect()
        } else {
            out.into_iter()
                .map(|x| if x.is_true() { x.into() } else { ().into() })
                .collect()
        }
    } else if out.is_list() {
        let out = out.as_list().unwrap();
        if out.len() != data.len() {
            return Err("out must have the same length as data".into());
        }
        out.into_iter().map(|(_, x)| x).collect()
    } else {
        return Err("out must be a character vector, logical vector, or list".into());
    }
    .into_iter()
    .map(|x| {
        if x.is_null() {
            Nullable::Null
        } else {
            Nullable::NotNull(x)
        }
    });
    let data = data.into_iter().zip(v).collect::<Vec<_>>();
    let v = parallelize(data, move |_, (data, out)| {
        maybe_mutating_return_matrix(data, out, &f)
    })?;
    if v.len() == 1 && !is_list {
        Ok(v.into_iter().next().unwrap())
    } else {
        Ok(v.into_iter().collect::<List>().into_robj())
    }
}

pub fn from_to_file(from_file: &str, from: &str, to: &Path, file_type: Option<&str>) -> PathBuf {
    let to_file = to.join(
        from_file
            .strip_prefix(from)
            .unwrap()
            .trim_matches('/')
            .trim_matches('\\'),
    );
    if let Some(file_type) = file_type {
        let parts = to_file
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split('.')
            .collect::<Vec<_>>();
        let mut keep = parts.len() - 1;
        if parts[parts.len() - 1] == "gz" {
            keep -= 1;
        }
        to_file
            .with_file_name(parts.into_iter().take(keep).collect::<Vec<_>>().join("."))
            .with_extension(file_type)
    } else {
        to_file
    }
}

pub fn list_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
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

#[derive(Debug)]
pub enum Mat {
    Ref(ExternalPtr<Mat>),
    Own(lmutils::Matrix),
}

impl Mat {
    pub fn ptr(&mut self) -> ExternalPtr<Self> {
        match self {
            Mat::Ref(r) => r.as_robj().clone().try_into().unwrap(),
            m @ Mat::Own(_) => {
                let slf = std::mem::replace(
                    m,
                    Mat::Own(lmutils::Matrix::Owned(OwnedMatrix::new(0, 0, vec![], None))),
                );
                let ptr = ExternalPtr::new(slf);
                ptr.clone().into_robj().set_class(&["Mat"]).unwrap();
                *m = Mat::Ref(ptr);
                m.ptr()
            }
        }
    }
}

impl IntoMatrix for Mat {
    fn into_matrix(self) -> lmutils::Matrix {
        match self {
            r @ Mat::Ref(_) => lmutils::Matrix::from_deref(r),
            Mat::Own(r) => r,
        }
    }
}

impl Deref for Mat {
    type Target = lmutils::Matrix;

    fn deref(&self) -> &Self::Target {
        match self {
            Mat::Ref(r) => r,
            Mat::Own(r) => r,
        }
    }
}

impl DerefMut for Mat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Mat::Ref(r) => &mut *r,
            Mat::Own(r) => r,
        }
    }
}
