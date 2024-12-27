#!/bin/bash
xmake f -c -p linux --toolchain=gcc-13 -a x86_64 -k binary -m release -vy -o build-linux &&\
xmake build -vy