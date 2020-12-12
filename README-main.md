# SCC 403 Coursework
### Kieran Molloy
These are in depth instructions on how to execute the results obtained in the paper submitted for the SCC403 module coursework component.

## SCC 403 - Coursework - Windows Section

### Installation
Using git clone within bash or alternative
<pre>
git clone https://github.com/K-Molloy/scc403-cw-windows.git
</pre>

### Cluster Analysis
To run cluster analysis, use R Studio to open `cluster_analysis.Rmd` or to see results open `cluster_analysis.nb.html`

### Python Analysis
To run the pre-processing and clustering, install all required modules using 
<pre>
pip install -r "requirements.txt"
</pre>
then run `main.py` and it will produce all graphs.
Ensure there is a /csv folder also a ../data folder



## SCC 403 - Coursework - Linux Section

### Installation
The script itself needs no installation, just copy it with the rest of the files in your working directory.
Alternatively you could use git clone
<pre>
sudo apt-get update && sudo apt-get install git && git clone https://github.com/K-Molloy/scc403-cw-linux.git
</pre>

### Happy path installation on Ubuntu 18.04LTS
<pre>
sudo apt-get update && sudo apt-get install git gcc build-essential swig python-pip virtualenv python3-dev
git clone https://github.com/K-Molloy/scc403-cw-linux.git
pip install virtualenv
virtualenv venv -p /usr/bin/python3.6
source zeroconf/bin/activate
curl https://raw.githubusercontent.com/K-Molloy/scc403-cw-linux/master/requirements.txt | xargs -n 1 -L 1 pip install
git clone https://github.com/K-Molloy/scc403-cw-linux.git
cd scc403-cw-linux/ && python ./bin/zeroconf.py -d ./data/df_final.h5 > results.txt
</pre>
