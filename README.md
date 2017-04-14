# Scientific Impact Prediction

This package computes citation predictions for papers and h-index predictions for authors using a dataset underlying the [Semantic Scholar](https://www.semanticscholar.org/) search service created by [AI2](http://allenai.org/). Here we describe how to acquire the data to train a collection of regression models and produce a collection of plots showing their relative performance.

If you would like to simply access the data and are uninterested in running the code then skip to the "Getting the data" section. Otherwise you will want to first clone this repository to a local directory:

```bash
git clone git@github.com:Lucaweihs/impact-prediction.git
```

### Table of Contents
1. [Getting the data](#getting-data)
2. [Code dependecies](#code-dependencies)
3. [Features used in prediction](#features)


## Getting the data <a name="getting-data"></a>

Data can be downloaded manually as individual files or, if you are just interested in producing predictions, just those files necessary to train models and produce predictions can be automatically downloaded and extracted using the download_data.py script. To use the download_data.py script run the commands:

```bash
# Enter the impact prediction directory
cd path/to/impact-prediction
# Run the script to download the data
python download_data.py
```

We now describe the individual files and provide URLs to download them manually.

### Data file descriptions

These data span the years between 1975 and 2015. The features are generated using information available only in 2005 and we train models to predict in the years 2006-2015. The data comes in two formats, tab separated values files (.tsv) and json files (.json); all data are compressed with gzip so be sure to unzip them (gzip -d filename). Authors are identified by their name and papers by a unique identifier. The paper ids correspond to those used by [Semanic Scholar](https://www.semanticscholar.org/) and more information about a paper with a given paper id can be found by using Semantic Scholar. For example, one can find more information about the paper with id

214899d16f39a494c3e69118c53a7b5877c0bbfc

by going to the URL:

[www.semanticscholar.org/paper/214899d16f39a494c3e69118c53a7b5877c0bbfc](https://www.semanticscholar.org/paper/214899d16f39a494c3e69118c53a7b5877c0bbfc).

#### Author names

*File name:*
authors-1975-2005-2015-2.tsv

*Format:*
Every line is the name of an author taken from a paper.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/authors-1975-2005-2015-2.tsv.gz)

#### Author features

*File name:*
authorFeatures-1975-2005-2015-2.tsv

*Format:*
The first line specifies the feature names and every other line represents the feature values for a particular author. These features are ordered to correspond to the authors from the "author names" file.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/authorFeatures-1975-2005-2015-2.tsv.gz)

#### Author responses

*File name:*
authorResponses-1975-2005-2015-2.tsv

*Format:*
The first line specfies the column names, and are of the form total_citations_in_NUMBER or hindex_in_NUMBER. Here NUMBER is the number of years since 2006 so that if a particular row has a value of 5 in the hindex_in_5 column this means that the author corresponding to that row had an hindex of 5 in year 2006 + 5 = 2011. These responses are ordered to correspond to the authors from the "author names" file.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/authorResponses-1975-2005-2015-2.tsv.gz)

#### Author histories

*File name:*
authorHistories-hindex-1975-2005-2015-2.tsv

*Format:*
Every line corresponds to the h-index of an author since the beginning of their career until 2005. These histories are ordered to correspond to the authors from the "author names" file.

*Example:*
If an author has a 5 year old career, by 2005, and their per-year h-index is 1,1,2,3,4. Then the line corresponding to said author would be
1 1 2 3 4

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/authorHistories-hindex-1975-2005-2015-2.tsv.gz)

#### Paper ids

*File name:*
paperIds-1975-2005-2015-2.tsv

*Format:*
Each line corresponds to a single paper id. These ids are unique identifiers of papers.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/paperIds-1975-2005-2015-2.tsv.gz)

#### Paper features

*File name:*
paperFeatures-1975-2005-2015-2.tsv

*Format:*
The first line specifies the feature names and every other line represents the feature values for a particular paper. These features are ordered to correspond to the papers from the "paper ids" file.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/paperFeatures-1975-2005-2015-2.tsv.gz)

#### Paper responses

*File name:*
paperResponses-1975-2005-2015-2.tsv

*Format:*
Each line corresponds to the observed cumulative citation count of an author in the years between 2006 and 2015. These responses are ordered to correspond to the paper from the "paper ids" file.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/paperResponses-1975-2005-2015-2.tsv.gz)

#### Paper histories

*File name:*
paperHistories-1975-2005-2015-2.tsv

*Format:*
Same as for "author histories" but replacing authors with papers and the h-index with cumulative citation counts.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/paperHistories-1975-2005-2015-2.tsv.gz)

#### Citation graph

*File name:*
citationGraph-1975-2015.json

*Format:*
Each line corresponds to a json dictionary with the following fields:
* id - a paper id
* cites - a list of the paper ids cited by id

*Notes:*
The citation graph includes all papers published between 1975 and 2015.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/citationGraph-1975-2015.json.gz)

#### Key citation graph

*File name:*
keyCitationGraph-1975-2015.json

*Format:*
Exactly as for the "citation graph" file but only includes key citations between papers.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/keyCitationGraph-1975-2015.json.gz)

#### Paper meta data

*File name:*
paperIdToPaperData-1975-2015.json

*Format:*
Every line is a json dictionary corresponding to a single paper with the following fields:
* citations-in-year - a dictionary of where the keys are years and the values are the number of citations the paper received in a particular year.
* is-survey - true/false depending on whether or not the paper is a survey.
* year - the publication year.
* id - the paper's id.
* authors - a list of the papers authors.
* venue - the venue where the paper was published.

[Download link](https://s3-us-west-2.amazonaws.com/ai2-s2/lucaw/paperIdToPaperData-1975-2015.json.gz)

## Code dependencies <a name="code-dependencies"></a>

This project is written in Python 2.7.11 using the following packages, divided into several categories.

*Data reading, representation, serialization:*
- cPickle
- configparser
- gzip
- pandas
- subprocess

*Parallel processing:*
- joblib
- multiprocessing

*Modeling/mathematics:*
- numpy
- scipy
- sklearn
- tensorflow

*Plotting:*
- matplotlib
- seaborn

The majority of these come preinstalled on, or can be easily installed through, any scientific python manager, e.g. [anaconda](https://www.continuum.io/downloads). Installing [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) requires some manual labor but is well described in the previous link.

## Training and comparing models

We should note that the training process can take many hours even on a strong machine, be prepared. That said, after a model has finished training we save the results so that you do not have to train it again. We assume you have downloaded the data using the download_data.py script described in the "Getting the data" section. To train a collection of models for h-index prediction and produce a collection of associated plots (placed in the ./plots directory) you can run the following command from within the impact-prediction directory:

```bash
python author_predictions.py hindex "author_hindex:4;author_age:5,12"
```

The created plots show the MAPE, R^2, and PA-R^2 metrics of the various trained algorithms on training, validation, and testing datasets. These plots are named to be self-descriptive. The above code trains and tests only on those authors with an h-index >= 4 by 2005 and whose career length was between 5 and 12 years in 2005.

To train models for paper citation prediction you can run the command:

```bash
python paper_predictions.py citation "paper_citations:5"
```

As above this will create a number of plots in the "plots" directory. Here the above command will only include those papers with >= 5 citations by 2005.

## Features used in prediction <a name="features"></a>

We have two sets of features used in our predictions, one for authors
and one for papers. The features are listed below.

### Author Features

|Feature Name | Description |
| ----------- | ----------- |
author\_hindex | H-index
author\_hindex\_delta | Change in h-index over the last 2 years
author\_citation\_count | Cumulative citation count
author\_key\_citation\_count | Cumulative key citation count (see Zhu et al. 2015)
author\_citations\_delta\_\{0,1\} |  Citations this year and 1 year ago
author\_key\_citations\_delta\_\{0,1\} | Key citations this year and 1 year ago
author\_mean\_citations\_per\_paper | Mean number of citations per paper
author\_mean\_citation\_per\_paper\_delta | Change in mean number of citations per paper over last 2 years
author\_mean\_citations\_per\_year | Mean number of citations per year
author\_papers | Number of papers published
author\_papers\_delta | Number of papers published in last 2 years
author\_mean\_citation\_rank | Rank of author (between 0 and 1) among all other authors in terms of mean citations per year
author\_unweighted\_pagerank | PageRank of author in unweighted coauthorship network
author\_weighted\_pagerank | PageRank of author in weighted coauthorship network
author\_age | Career length (years since first paper published)
author\_recent\_num\_coauthors | Total number of coauthors in last 2 years
author\_max\_single\_paper\_citations | Max number of citations for any of author's papers
venue\_hindex\_\{mean, min,max\} | H-indices of venues author has published in
venue\_hindex\_delta\_\{mean, min,max\} | Change in h-index over last two years for venues author has published in
venue\_citations\_\{mean, min,max\} | Mean citations per paper of venues author has published in
venue\_citations\_delta\_\{mean, min,max\} | Change in mean citations per paper over last two years for venues author has published in
venue\_papers\_\{mean, min, max\} | Number of papers in venues in which the author has published
venue\_papers\_delta\_\{mean, min, max\}  | Change in number of papers in venues in which  the author has published over the last 2 years
venue\_rank\_\{mean, min, max\} | Ranks of venues (between 0-1) in which the author has published determined by mean number of citations per paper
venue\_max\_single\_paper\_citations\_\{mean, min, max\} | Maximum number of citations any paper published  in a venue has received for each venue the author has published in
total\_num\_venues | Total number of venues published in


### Paper Features


|Feature Name | Description |
| ----------- | ----------- |
author\_hindex\_\{mean, min, max\} | H-indices of authors
author\_hindex\_delta\_\{mean, min, max\} | Change in h-indices of authors in last 2 years
author\_citations\_\{mean, min, max\} | Cumulative citations for each author
author\_citations\_delta\_\{mean, min, max\} | Change in cumulative citations for each author in last 2 years
author\_key\_citations\_\{mean, min, max\} | Cumulative key citations for each author
author\_key\_citations\_delta\_\{mean, min, max\} | Change in cumulative key citations for each author in last 2 years
author\_mean\_citations\_\{mean, min, max\} | Mean citations per paper for each author
author\_mean\_citations\_delta\_\{mean, min, max\} | Change in mean citations per paper for each author in last 2 years
author\_papers\_\{mean, min, max\} | Number of papers published for each author
author\_papers\_delta\_\{mean, min, max\} | Number of papers published for each author in last 2 years
author\_unweighted\_pagerank\_\{mean, min, max\} | PageRank of each author in the unweighted coauthorship network
author\_weighted\_pagerank\_\{mean, min, max\} | PageRank of each author in the weighted coauthorship network
author\_mean\_citation\_rank\_\{mean, min, max\} | Rank of each author among all authors in terms of mean citations per paper
author\_recent\_num\_coauthor\_\{mean, min, max\} | Number of coauthors each author had in last 2 years
author\_max\_single\_paper\_citations\_\{mean, min, max\} | Maximum citations a single paper of each author has received
total\_num\_authors | Total number of authors for the paper
venue\_hindex | H-index of the venue
venue\_hindex\_delta | Change in h-index of the venue in last 2 years
venue\_mean\_citations | Mean citations per paper published in the venue
venue\_mean\_citations\_delta | Change in mean citations per paper published in the venue in last 2 years
venue\_papers | Number of papers published in the venue
venue\_papers\_delta | Number of papers published in the venue in last 2 years
venue\_rank | Rank of the venue among all venues in terms of mean citations per paper
venue\_max\_single\_paper\_citations | Maximum number of citations any paper published in the venue has received
paper\_age | Age of the paper in years (rounded up)
paper\_citations | Cumulative citation count
paper\_key\_citations | Cumulative key citation count
paper\_mean\_citations\_per\_year | Average number of citations received per year
is\_survey | Whether or not the paper is a survey
paper\_citations\_delta\_\{0,1\} | Number of citations the paper received in the last year and the year before that
paper\_key\_citations\_delta\_\{0,1\} | Number of key citations the paper received in the last year and the year before that
