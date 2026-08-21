# Build the manuscript with XeLaTeX. Fonts load by filename through kpathsea,
# so no system font registration is required.
PAPER = few

.PHONY: all clean watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(PAPER).tex

watch:
	latexmk -xelatex -pvc -interaction=nonstopmode $(PAPER).tex

clean:
	latexmk -C
	rm -f $(PAPER).bbl $(PAPER).blg
