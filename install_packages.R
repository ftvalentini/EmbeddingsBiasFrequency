

urls = c(
    "https://cran.r-project.org/src/contrib/tidyverse_1.3.1.tar.gz",
    "https://cran.r-project.org/src/contrib/rmarkdown_2.14.tar.gz",
    "https://cran.r-project.org/src/contrib/scales_1.2.0.tar.gz",
    "https://cran.r-project.org/src/contrib/glue_1.6.2.tar.gz",
    "https://cran.r-project.org/src/contrib/latex2exp_0.9.4.tar.gz",
    "https://cran.r-project.org/src/contrib/cowplot_1.1.1.tar.gz",
    "https://cran.r-project.org/src/contrib/remotes_2.4.2.tar.gz"
)


for (url in urls) {
    install.packages(url, repos=NULL, type="source", dependencies=T)
}


remotes::install_github(
    "rpkgs/gg.layers", ref="c670082b356d92864e92ec297c1a58fcb649e4b9")
# NOTE might require:
# sudo apt-get install libfftw3-dev libfftw3-doc
