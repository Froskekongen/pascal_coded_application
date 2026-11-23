# Overleaf compiles from the repository root. Resolve the canonical paper's local
# class, inputs and figure paths while retaining the standard TeX paths.
$ENV{'TEXINPUTS'} = './paper/source//:' . ($ENV{'TEXINPUTS'} // '');
