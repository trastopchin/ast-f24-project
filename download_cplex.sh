#!/usr/bin/env bash

CPLEX_URL="https://ak-dsw-mul.dhe.ibm.com/sdfdl/v2/fulfill/M08T0ML/Xa.2/Xb.XwdHXFnVukQ7nmOw3Ek/Xc.M08T0ML/cplex_studio2211.linux_x86_64.bin/Xd./Xf.lPr.D1VC/Xg.12798813/Xi./XY.scholars/XZ.FEs3Q0GMbraoCBvrUs6GC06ykqvGOnbL/cplex_studio2211.linux_x86_64.bin"

# Only download if it doesn't exist
[ -f cplex_studio2211.linux_x86_64.bin ] || wget -O cplex_studio2211.linux_x86_64.bin "$CPLEX_URL"