import pickle, sys, numpy as np
sys.path.insert(0, "/scratch/st1084601/thesis")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

c = pickle.load(open("/scratch/st1084601/thesis/outputs/hybrid_cache_official_v2.pkl", "rb"))
D = {k: c[k + "_dataset"] for k in ("train", "validation", "test")}


def get(ds):
    return (np.asarray(ds.features, dtype=np.float32), np.asarray(ds.labels),
            np.asarray(ds.subjects), np.asarray(ds.trials))


def subj_norm(X, S):
    X = X.copy()
    for s in np.unique(S):
        m = S == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return X


def trial_f1(p, y, S, T):
    keys = {}
    for pi, yi, si, ti in zip(p, y, S, T):
        keys.setdefault((si, ti), []).append((pi, yi))
    pp = [np.mean([a for a, _ in v]) for v in keys.values()]
    yy = [v[0][1] for v in keys.values()]
    return pp, yy


tr, va, te = get(D["train"]), get(D["validation"]), get(D["test"])
print("feature dim", tr[0].shape, "train windows", len(tr[1]))
for name, sl in (("all278", slice(0, 278)), ("eeg256", slice(0, 256)), ("physio22", slice(256, 278))):
    for norm in (False, True):
        res = []
        for C in (0.01, 0.1):
            Xs = [subj_norm(d[0][:, sl], d[2]) if norm else d[0][:, sl] for d in (tr, va, te)]
            clf = LogisticRegression(C=C, max_iter=300).fit(Xs[0], tr[1])
            out = []
            for X, d in ((Xs[1], va), (Xs[2], te)):
                pp, yy = trial_f1(clf.predict_proba(X)[:, 1], d[1], d[2], d[3])
                out.append(f1_score(yy, (np.array(pp) > 0.5).astype(int), average="macro"))
            res.append((C, round(out[0], 3), round(out[1], 3)))
        print(f"{name:9s} subjnorm={norm!s:5s} (C, valTrialF1, testTrialF1): {res}", flush=True)
