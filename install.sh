export TMPDIR=$(mktemp -d --tmpdir=$(pwd))
# install rust if not installed
if [ -f $HOME/.cargo/env ]; then
    echo "LMUTILS:: Rust is already installed"
    rustup update > /dev/null
    echo "LMUTILS:: Rust is updated"
else
    echo "LMUTILS:: Rust not found, installing"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null
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
# make $HOME/.lmutils if not exists
if [ ! -d $HOME/.lmutils ]; then
    mkdir $HOME/.lmutils
fi
R -e "install.packages('https://github.com/GMELab/lmutils.r/releases/latest/download/lmutils.r.tar.gz', repos = NULL, lib = '$HOME/.lmutils')" > /dev/null
if [ $? -eq 0 ]; then
    echo "LMUTILS:: lmutils is now installed"
else
    echo "LMUTILS:: lmutils installation FAILED"
    rm -rf $TMPDIR
    unset TMPDIR
    exit 1
fi
# if not exists, add $HOME/.lmutils to .libPaths()
if [ ! -f $HOME/.Rprofile ]; then
    touch $HOME/.Rprofile
fi
if ! grep -q "$HOME/.lmutils" $HOME/.Rprofile; then
    echo ".libPaths(c('$HOME/.lmutils', .libPaths()))" >> $HOME/.Rprofile
fi
rm -rf $TMPDIR
unset TMPDIR
