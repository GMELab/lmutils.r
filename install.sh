# if -v is passed, then verbose mode is enabled (set a variable to print commands)
verbose=0
while getopts 'v' flag; do
    case "${flag}" in
        v) verbose=1 ;;
        *) error "Unexpected option ${flag}" ;;
    esac
done

export TMPDIR=$(mktemp -d --tmpdir=$(pwd))
# install rust if not installed
if [ -f $HOME/.cargo/env ]; then
    echo "LMUTILS:: Rust is already installed"
    rustup update
    echo "LMUTILS:: Rust is updated"
else
    echo "LMUTILS:: Rust not found, installing"
    if [ $verbose -eq 1 ]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null
    fi
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
if [ $verbose -eq 1 ]; then
    R -e "install.packages('https://github.com/GMELab/lmutils.r/releases/latest/download/lmutils.r.tar.gz', repos = NULL, lib = '$HOME/.lmutils')"
else
    R -e "install.packages('https://github.com/GMELab/lmutils.r/releases/latest/download/lmutils.r.tar.gz', repos = NULL, lib = '$HOME/.lmutils')" > /dev/null
fi
if [ $? -eq 0 ]; then
    echo "LMUTILS:: lmutils is now installed"
else
    echo "LMUTILS:: lmutils installation FAILED"
    rm -rf $TMPDIR
    unset TMPDIR
    exit 1
fi
# if not exists, add $HOME/.lmutils to .libPaths()
if ! grep -q "$HOME/.lmutils" $HOME/.Rprofile; then
    echo ".libPaths(c('$HOME/.lmutils', .libPaths()))" >> $HOME/.Rprofile
fi
rm -rf $TMPDIR
unset TMPDIR
