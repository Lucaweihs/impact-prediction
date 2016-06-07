import urllib
import os

def download_and_extract(file_name):
    print("Downloading " + file_name)
    urllib.urlretrieve("https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/" + file_name + ".gz",
                       "data/" + file_name + ".gz")
    os.system("gzip -d data/" + file_name + ".gz")

download_and_extract("authors-1975-2005-2015-2.tsv")
download_and_extract("authorFeatures-1975-2005-2015-2.tsv")
download_and_extract("authorHistories-hindex-1975-2005-2015-2.tsv")
download_and_extract("authorResponses-1975-2005-2015-2.tsv")

download_and_extract("paperIds-1975-2005-2015-2.tsv")
download_and_extract("paperFeatures-1975-2005-2015-2.tsv")
download_and_extract("paperHistories-1975-2005-2015-2.tsv")
download_and_extract("paperResponses-1975-2005-2015-2.tsv")


