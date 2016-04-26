#!/bin/sh

for figure in $(ls fig/*.dia)
do
    dia -t eps -l -O fig/ $figure
done

latex *.tex
dvipdf *.dvi
