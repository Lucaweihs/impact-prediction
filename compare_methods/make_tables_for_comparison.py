import pandas as pd
import re

def printLatexTable(df):
    table = df[['2006', '2010', '2015']].apply(lambda x: pd.Series.round(x, 3)).to_latex()
    table = re.sub(r"toprule|midrule|bottomrule", "hline", table)
    table = re.sub(r"lrrr", "|l|rrr|", table)
    print table

with open("tables/inflated-rsq-test-author-hindex-authorAge:5,12-authorHInds:4-1975-2005-2015-2.tsv", 'rb') as f:
    hIndexRsqDf = pd.read_csv(f, sep = "\t", index_col=0)

printLatexTable(hIndexRsqDf)

with open("tables/inflated-rsq-test-author-citation-authorAge:5,12-authorHInds:4-1975-2005-2015-2.tsv", 'rb') as f:
    authorCitationRsqDf = pd.read_csv(f, sep = "\t", index_col=0)

printLatexTable(authorCitationRsqDf)