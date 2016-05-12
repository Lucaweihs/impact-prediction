import pandas as pd

with open("tables/inflated-rsq-valid-author-hindex-0-0-4-1975-2005-2015-2.tsv", 'rb') as f:
    hIndexRsqDf = pd.read_csv(f, sep = "\t", index_col=0)

print hIndexRsqDf[['2006', '2010', '2015']].apply(lambda x: pd.Series.round(x, 2)).to_latex()

with open("tables/inflated-rsq-valid-author-citation-5-0-1975-2005-2015-2.tsv", 'rb') as f:
    authorCitationRsqDf = pd.read_csv(f, sep = "\t", index_col=0)

print authorCitationRsqDf[['2006', '2010', '2015']].apply(lambda x: pd.Series.round(x, 2)).to_latex()