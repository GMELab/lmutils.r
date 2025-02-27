echo "LMUTILS:: Installing Rust"
export TMPDIR=$(mktemp -d --tmpdir=$(pwd))
# install rust if not installed
if [ -f $HOME/.cargo/env ]; then
    echo "LMUTILS:: Rust is already installed"
    rustup update
    echo "LMUTILS:: Rust is updated"
else
    echo "LMUTILS:: Rust not found, installing"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    if [ -f $HOME/.cargo/env ]; then
        echo "LMUTILS:: Rust is now installed"
    else
        echo "LMUTILS:: Rust installation FAILED"
        rm -rf $TMPDIR
        unset TMPDIR
        exit 1
    fi
fi

echo "LMUTILS:: Installing lmutils"
R -e "install.packages('https://github.com/GMELab/lmutils.r/releases/latest/download/lmutils.r.tar.gz', repos = NULL)" > /dev/null
if [ $? -eq 0 ]; then
    echo "LMUTILS:: lmutils is now installed"
else
    echo "LMUTILS:: lmutils installation FAILED"
    rm -rf $TMPDIR
    unset TMPDIR
    exit 1
fi
rm -rf $TMPDIR
unset TMPDIR
