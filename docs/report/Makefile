PROJECT := group5_Report
.PHONY: all pdf clean
all: pdf
pdf:
	cd main && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../.build main.tex
	cp .build/main.pdf $(PROJECT).pdf
clean:
	cd main && latexmk -C -outdir=../.build main.tex
	rm -rf .build
