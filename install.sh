echo "LMUTILS:: Installing Rust"
export TMPDIR=$(mktemp -d --tmpdir=$(pwd))
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
unset TMPDIR
if [ -f $HOME/.cargo/env ]; then
    echo "LMUTILS:: Rust is installed"
else
    echo "LMUTILS:: Rust is not installed"
fi

echo "LMUTILS:: Installing lmutils"
R -e "install.packages('https://github.com/GMELab/lmutils.r/releases/download/latest/lmutils.r.tar.gz', repos = NULL)" > /dev/null
if [ $? -eq 0 ]; then
    echo "LMUTILS:: lmutils is installed"
else
    echo "LMUTILS:: lmutils is not installed"
fi
