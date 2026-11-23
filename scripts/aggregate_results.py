#!/usr/bin/env python3
"""Produce transparent comparisons from actual completed Kaggle experiment files."""
from pathlib import Path
import argparse,json,csv,statistics,collections
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,help="Downloaded Kaggle results root")
    args=parser.parse_args()
    root=Path(args.source)
    if not root.is_dir():raise SystemExit("Missing results directory")
    files=list(root.rglob("paper_*_completed.json"))
    records=[]
    for f in files:
        obj=json.loads(f.read_text(encoding="utf-8"))
        m=obj.get("metrics",{})
        if m and m.get("model") and m.get("seed") is not None:
            records.append({k:v for k,v in m.items() if isinstance(v,(int,float,str))})
    print("VALID_COMPLETION_MARKERS",len(records))
    if not records:raise SystemExit("No completed paper-mode runs. No metrics can be reported.")
    dst=root/"validated_comparison.csv"
    fields=sorted(set(k for record in records for k in record))
    with dst.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(records)
    grouped=collections.defaultdict(list)
    for r in records:grouped[r["model"]].append(float(r["macro_f1"]))
    for model,vals in grouped.items():
        print(model,"N",len(vals),"macro-F1_mean",statistics.mean(vals),
            "macro-F1_std",statistics.stdev(vals) if len(vals)>=2 else None)
    print("OUTPUT",dst)
if __name__=="__main__":main()
