"""Paired group bootstrap: same held-out series/near-duplicate groups in both arms."""
import numpy as np
from sklearn.metrics import f1_score

def paired_group_bootstrap(baseline, proposed, group_key="group", replicates=2000, seed=2026):
    required=["relative_path","label","pred",group_key]
    for frame in (baseline,proposed):
        assert set(required).issubset(frame.columns),set(required)-set(frame.columns)
    assert list(baseline["relative_path"])==list(proposed["relative_path"])
    assert np.array_equal(baseline["label"].to_numpy(),proposed["label"].to_numpy())
    assert np.array_equal(baseline[group_key].to_numpy(),proposed[group_key].to_numpy())
    true=baseline["label"].to_numpy(dtype=int)
    b=baseline["pred"].to_numpy(dtype=int)
    p=proposed["pred"].to_numpy(dtype=int)
    groups=baseline[group_key].to_numpy()
    uniq=np.unique(groups)
    assert len(uniq)>=2,"Need more than one independent group"
    positions={g:np.where(groups==g)[0] for g in uniq}
    def f(y,z):return f1_score(y,z,labels=[0,1],average="macro",zero_division=0)
    delta=float(f(true,p)-f(true,b))
    rng=np.random.default_rng(seed)
    draws=np.empty(replicates,dtype=float)
    for k in range(replicates):
        selected=rng.choice(uniq,size=len(uniq),replace=True)
        indices=np.concatenate([positions[g] for g in selected])
        draws[k]=float(f(true[indices],p[indices])-f(true[indices],b[indices]))
    lo,hi=np.quantile(draws,[0.025,0.975])
    return delta,float(lo),float(hi),len(uniq)
