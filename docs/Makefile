SOURCES = $(wildcard *.pmd)
INDIV_HTML_PAGES = $(SOURCES:.pmd=.html)
HTML_TARGETS = Training.html
PDF_TARGETS = Traininig.pdf
MD_TARGETS = $(SOURCES:.pmd=.md)

%.md : %.pmd
	pweave -f pandoc -o $@ $<

%.html : %.pmd
	pweave -f pandoc2html -o $@ $<

Training.html: $(MD_TARGETS)
	pandoc --css pweave.css --toc -t html -o Training.html *.md

Training.pdf: $(MD_TARGETS)
	pandoc --template eisvogel.tex  -o Training.pdf --toc -N --highlight-style tango  *.md

md : $(MD_TARGETS)

html: Training.html
pdf: Training.pdf

indiv_html : $(INDIV_HTML_PAGES)


clean:
	rm *.log *.aux *.pdf *.out *.html
