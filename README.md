# Scientific Impact Prediction

## Introduction

This package computes citation predictions for papers and h-index predictions for authors using a dataset underlying the [Semantic Scholar](https://www.semanticscholar.org/) search service created by [AI2](http://allenai.org/). Here we describe how to acquire the data to train a collection of regression models and produce a collection of plots showing their relative performance. 

If you would like to simply access the data and are uninterested in running the code then skip to the "Getting the data" section. Otherwise you will want to first clone this repository to a local directory:

```bash
git clone git@github.com:Lucaweihs/impact-prediction.git
```

## Getting the data

Data can be downloaded manually as individual files or, if you are just interested in producing predictions, just those files necessary to train models and produce predictions can be automatically downloaded using the download_data.py script. To use the download_data.py script run the commands:

```bash
# Enter the impact prediction directory
cd path/to/impact-prediction
# Run the script to download the data
python download_data.py
```

We now describe the individual files and provide URLs to download them manually.

### Data file descriptions

These data span the years between 1975 and 2015. The features are generated using information available only in 2005 and we train models to predict in the years 2006-2015. The data comes in two formats, tab separated values files (.tsv) and json files (.json).

#### Author names

*File name:*
authors-1975-2005-2015-2.tsv

*Format:*
Every line is the name of an author taken from a paper.

*URL:*

#### Author features

*File name:*
authorFeatures-1975-2005-2015-2.tsv

*Format:*
The first line specifies the feature names and every other line represents the feature values for a particular author. These features are ordered to correspond to the authors from the "author names" file.

*URL:*

#### Author responses

*File name:*
authorResponses-1975-2005-2015-2.tsv

*Format:*
Each line corresponds to the observed h-index of an author in the years between 2006 and 2015. These responses are ordered to correspond to the authors from the "author names" file.

*URL:*

#### Author histories

*File name:*
authorHistories-hindex-1975-2005-2015-2.tsv

*Format:*
Every line corresponds to the h-index of an author since the beginning of their career until 2005. These histories are ordered to correspond to the authors from the "author names" file.

*URL:*

*Example:*
If an author has a 5 year old career, by 2005, and their per-year h-index is 1,1,2,3,4. Then the line corresponding to said author would be
1 1 2 3 4
 
#### Paper ids

*File name:*
paperIds-1975-2005-2015-2.tsv

*Format:*
Each line corresponds to a single paper id. These ids are unique identifiers of papers.

*URL:*

#### Paper features

*File name:*
paperFeatures-1975-2005-2015-2.tsv

*Format:*
The first line specifies the feature names and every other line represents the feature values for a particular paper. These features are ordered to correspond to the papers from the "paper ids" file.

*URL:*

#### Paper responses

*File name:*
paperResponses-1975-2005-2015-2.tsv

*Format:*
Each line corresponds to the observed cumulative citation count of an author in the years between 2006 and 2015. These responses are ordered to correspond to the paper from the "paper ids" file.

*URL:*

#### Paper histories

*File name:*
paperHistories-1975-2005-2015-2.tsv

*Format:*
Same as for "author histories" but replacing authors with papers and the h-index with cumulative citation counts.

*URL:*

#### Citation graph

*File name:*
citationGraph-1975-2015.tsv

*Format:*
Each line corresponds to a json dictionary with the following fields:
* id - a paper id
* cites - a list of the paper ids cited by id

*URL:*

*Notes:*
The citation graph includes all papers published between 1975 and 2015.

#### Key citation graph

*File name:*
keyCitationGraph-1975-2015.tsv

*Format:*
Exactly as for the "citation graph" file but only includes key citations between papers.

*URL:*

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

*URL:*

## Code Dependencies

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

The majority of these come preinstalled on, or can be easily install through, any scientific python manager, e.g. [anaconda](https://www.continuum.io/downloads). Installing [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) requires some manual labor but is well described in the previous link.

## Training and comparing models

We assume you have downloaded the data using the download_data.py script described in the "Getting the data" section. To train a collection of models for h-index prediction and produce a collection of associated plots you can run the following command from within the impact-prediction directory:

```bash
python author_predictions hindex author_hindex:4;author_age:5,12
```

Beyond training models, this will produce a number of plots in the "plots" directory; these plots show the MAPE, R^2, and PA-R^2 metrics of the various trained algorithms on a training, validation, and testing datasets. These plots are named to be self-descriptive. The above code trains and tests only on those authors with an h-index >= 4 by 2005 and whose career length was between 5 and 12 years in 2005.
 
To train models for paper citation prediction you can run the command:
 
```bash
python paper_predictions citation paper_citations:5
```

As above this will create a number of plots in the "plots" directory. Here the above command will only include those papers with >= 5 citations by 2005.