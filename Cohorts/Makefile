SOURCES=$(shell ls *.Rmd)
TARGETS=$(SOURCES:%.Rmd=%.html)


%.html: %.Rmd
	@echo "$< -> $@"
	@Rscript -e "rmarkdown::render('$<', output_format='html_document')"

default: $(TARGETS)