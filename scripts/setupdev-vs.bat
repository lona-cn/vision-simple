xmake f -c -p windows -a x64 -m "releasedbg" --runtimes="MDd" --with_dml=true --with_cuda=false -vDy &&^
xmake project -k vsxmake -m "releasedbg" -a "x64" -P . -y -vD