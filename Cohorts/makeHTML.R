library(rmarkdown)

rmdfiles = dir(pattern='Rmd')
for(f in rmdfiles){
  render(f, output_format='html_document')
}