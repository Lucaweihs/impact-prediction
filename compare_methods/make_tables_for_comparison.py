import pandas as pd
import re

def print_latex_table(df):
    table = df[['2006', '2010', '2015']].apply(lambda x: pd.Series.round(x, 3)).to_latex()
    table = re.sub(r"toprule|midrule|bottomrule", "hline", table)
    table = re.sub(r"lrrr", "|l|rrr|", table)
    print table

with open("tables/inflated-rsq-test-author-hindex-author_age:5,12-author_hindex:4-1975-2005-2015-2.tsv", 'rb') as f:
    h_index_rsq_df = pd.read_csv(f, sep = "\t", index_col=0)

print_latex_table(h_index_rsq_df)

with open("tables/inflated-rsq-test-author-citation-author_age:5,12-author_hindex:4-1975-2005-2015-2.tsv", 'rb') as f:
    author_citation_rsq_df = pd.read_csv(f, sep = "\t", index_col=0)

print_latex_table(author_citation_rsq_df)