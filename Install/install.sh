#!/usr/bin/env bash

# install R dependencies Ubuntu
apt-get update -y
apt-get upgrade -y
apt-get -y install apt-utils
apt-get -y install libssl-dev
apt-get -y install libxml2-dev
apt-get -y install libcairo2-dev
apt-get -y install libssh2-1-dev
apt-get -y install libcurl4-openssl-dev

# install devtools & largevis
R -e 'install.packages(c("devtools"), repos="http://cran.us.r-project.org", dependencies=TRUE)'
R -e "devtools::install_local('/largeVis-line')"

# install python requirements
pip install -r requirements.txt
