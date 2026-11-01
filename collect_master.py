import os, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import pandas as pd, numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
import helpers.data_mining_helpers as dmh
from helpers.feature_filtering import variance_filter, pearson_filter, spearman_filter

out = []
def P(*a):
    out.append(" ".join(str(x) for x in a))

cats = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
tt = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42)
X = pd.DataFrame.from_records(dmh.format_rows(tt), columns=['text'])
X['category'] = tt.target
X['category_name'] = X.category.apply(lambda t: dmh.format_labels(t, tt))

P("== Q1 (Section 3.2)")
P("sci.med documents:", len(X.loc[X['category_name'] == 'sci.med']), "| total documents:", len(X))

# 照 notebook 的順序：加入缺值的 dummy 再 dropna、加入重複的 dummy 再 drop_duplicates(keep=False)
X = pd.concat([X, pd.DataFrame([{'text': 'dummy_record', 'category': 1}])], ignore_index=True)
X.dropna(inplace=True)
X = pd.concat([X, pd.DataFrame([{'text': 'dummy record', 'category': 1, 'category_name': 'dummy category'}] * 2)], ignore_index=True)
X.drop_duplicates(keep=False, inplace=True)

P("== Q3 (Section 5.1, X.sample(n=1000, random_state=1) as in the notebook)")
Xs = X.sample(n=1000, random_state=1)
vc, vs = X.category_name.value_counts(), Xs.category_name.value_counts()
for c in vc.index:
    P(f"  {c:24s} original {vc[c]:4d} ({vc[c]/len(X)*100:5.2f}%) | sample {vs.get(c,0):4d} ({vs.get(c,0)/len(Xs)*100:5.2f}%)")

P("== Q4 (Section 5.3)")
for sw in ['english', None]:
    cv = CountVectorizer(stop_words=sw)
    M = cv.fit_transform(X.text)
    sp = 100 * (1 - M.nnz / (M.shape[0] * M.shape[1]))
    P(f"  stop_words={sw!r}: shape={M.shape}, non-zero={M.nnz}, sparsity={sp:.4f}%")

count_vect = CountVectorizer(stop_words='english')
X_counts = count_vect.fit_transform(X.text)
feature_names = count_vect.get_feature_names_out()

P("== Q5 (Section 5.3.1)")
for th in [0.0, 0.01, 0.05]:
    v = variance_filter(X_counts, feature_names, threshold=th)
    P(f"  threshold={th}: kept={int(v['keep'].sum())}, removed={int((~v['keep']).sum())}, total={len(v)}")
v = variance_filter(X_counts, feature_names, threshold=0.05)
rem = v[~v['keep']]
P("  example removed terms (lowest variance):", list(rem.tail(10)['feature']))
dfreq = np.asarray((X_counts > 0).sum(axis=0)).ravel()
low = v[v['variance'] < 0.01]['feature']
idx = {f: i for i, f in enumerate(feature_names)}
P(f"  doc-frequency of terms with variance<0.01: median={int(np.median([dfreq[idx[f]] for f in low]))} docs (of {X_counts.shape[0]})")

P("== Q6 (Section 5.3.2, target comp.graphics)")
y = (X["category_name"] == "comp.graphics").astype(int)
pdf = pearson_filter(X_counts, y, feature_names).head(10)
sdf = spearman_filter(X_counts, y, feature_names, max_terms=1500).head(10)
P("  Pearson top10:", [(r.feature, round(float(r.pearson_r), 4)) for r in pdf.itertuples()])
P("  Spearman top10:", [(r.feature, round(float(r.spearman_r), 4)) for r in sdf.itertuples()])

P("== Q7 (Section 5.4.2, term_freq transactional DB, category alt_atheism)")
cat = 'alt.atheism'
cv = CountVectorizer(stop_words='english')
Mc = cv.fit_transform(X[X['category_name'] == cat]['text'])
dfc = pd.DataFrame(Mc.toarray(), columns=cv.get_feature_names_out())
m = dfc.sum(axis=0)
lo, hi = np.percentile(m, 5), np.percentile(m, 95)
tf_df = dfc[m[(m >= lo) & (m <= hi)].index]
from PAMI.extras.convert.DF2DB import DF2DB
os.makedirs('./output_files/term_freq/transactional_dbs', exist_ok=True)
db = './output_files/term_freq/transactional_dbs/td_freq_db_alt_atheism.csv'
DF2DB(tf_df).convert2TransactionalDatabase(db, '>=', 1)
from PAMI.frequentPattern.topk import FAE as fae_alg
from PAMI.frequentPattern.maximal import MaxFPGrowth as maxfp_alg
for k in [100, 500]:
    o = fae_alg.FAE(iFile=db, k=k); o.mine()
    P(f"  Top-K FAE k={k}: patterns={len(o.getPatternsAsDataFrame())}, runtime={o.getRuntime():.3f}s")
for ms in [9, 3]:
    o = maxfp_alg.MaxFPGrowth(iFile=db, minSup=ms); o.mine()
    d = o.getPatternsAsDataFrame()
    longest = int(d['Patterns'].astype(str).str.split().str.len().max()) if len(d) else 0
    P(f"  MaxFPGrowth minSup={ms}: patterns={len(d)}, runtime={o.getRuntime():.3f}s, longest pattern={longest} items")

P("== Q10 (Section 5.6)")
lb = LabelBinarizer().fit(X.category)
P("  unique X['category'] values:", sorted(X.category.unique().tolist()), "| binarized shape:", lb.transform(X.category).shape)
P("  example:", X.category.iloc[0], "->", lb.transform(X.category.iloc[:1]).tolist()[0])

print("\n".join(out))
