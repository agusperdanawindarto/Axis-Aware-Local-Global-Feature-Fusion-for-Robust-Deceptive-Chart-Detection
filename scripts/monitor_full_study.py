#!/usr/bin/env python3
"""One-shot monitor: mirror finished Kaggle models and launch ablations only after nine valid results."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,datetime
PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sync_kaggle_results import check,command,EXPECTED_MANIFEST
MAIN="agusperdana/notebook38f0f2724f"
ABLATION="agusperdana/salf-dino-hvpd-branch-ablations"
STATUS=PROJECT/"local_artifacts"/"monitor_status.json"
STATUS.parent.mkdir(parents=True,exist_ok=True)
def update(**data):
    payload={"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**data}
    tmp=STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    os.replace(tmp,STATUS)
    print(json.dumps(payload,ensure_ascii=False),flush=True)
def download(version,kernel):
    target=PROJECT/"local_artifacts"/("kaggle_v"+str(version) if kernel==MAIN else "kaggle_ablation_v"+str(version))
    marker=target/"_sync_complete.json"
    if marker.exists():return target
    cmd=[sys.executable,str(PROJECT/"scripts"/"sync_kaggle_results.py"),
         "--kernel",kernel,"--version",str(version)]
    z=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    print("SYNC_OUTPUT",z.returncode,z.stdout[-1400:],z.stderr[-300:],flush=True)
    if z.returncode!=0 or not marker.exists():
        raise RuntimeError("Kaggle result sync incomplete (no complete marker)")
    return target
def collect(root):
    found=[]
    for p in root.rglob("paper_*_completed.json"):
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            metrics=d.get("metrics",{})
            found.append((metrics.get("model"),int(metrics.get("seed"))))
        except Exception as e:print("IGNORED_BAD_COMPLETION",p,str(e)[:150],flush=True)
    return set(found)
def verify_manifest(root):
    items=list(root.rglob("split_manifest.csv"))
    if len(items)!=1:raise RuntimeError("Missing or ambiguous split_manifest.csv")
    digest=hashlib.sha256(items[0].read_bytes()).hexdigest()
    if digest!=EXPECTED_MANIFEST:
        raise RuntimeError("Group-safe split checksum mismatch: "+digest)
    return digest
def push_ablations():
    D=ROOT/"KAGGLE"/"ablations_upload"
    meta=json.loads((D/"kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id")!=ABLATION or meta.get("is_private") is not True:
        raise RuntimeError("Ablation notebook id/private flag mismatch")
    nb=json.loads((D/meta["code_file"]).read_text(encoding="utf-8"))
    src="\n".join("".join(c["source"]) for c in nb["cells"])
    for t in ("RUN_ABLATIONS = True","MODE = 'paper'","SIX ABLATION RUNS COMPLETED",EXPECTED_MANIFEST):
        if t not in src:raise RuntimeError("Ablation notebook missing required guard: "+t)
    result=command(["kernels","push","-p",str(D),"--accelerator","NvidiaTeslaT4","-t","7200"],150)
    print("ABLATION_SUBMISSION",result[-800:],flush=True)
    import re
    match=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",result)
    if not match:raise RuntimeError("Unable to confirm ablation version ID")
    version=int(match.group(1))
    marker=PROJECT/"local_artifacts"/"ablation_launch.json"
    marker.write_text(json.dumps({"kernel":ABLATION,"version":version,"response":result},indent=2),
                      encoding="utf-8")
    return version
def main():
    main_status=check(7,MAIN)
    if main_status in ("RUNNING","QUEUED","PENDING"):
        update(stage="main",status=main_status,description="Waiting for Kaggle output to finalize")
        return
    if main_status not in ("COMPLETE","ERROR","CANCELLED"):
        update(stage="main",status=main_status,description="Unexpected Kaggle state")
        return
    root=download(7,MAIN)
    existing=collect(root)
    needed={(name,seed) for seed in (42,123,2025) for name in
            ("dinov3_global_avg","proposed_salf","swinv2_tiny")}
    missing=sorted(needed-existing)
    digest=verify_manifest(root)
    if missing:
        update(stage="main",status="incomplete",kaggle_status=main_status,
               completed=len(needed)-len(missing),missing=missing,
               split_manifest_sha256=digest)
        return
    launch=PROJECT/"local_artifacts"/"ablation_launch.json"
    if not launch.exists():
        version=push_ablations()
        update(stage="ablation",status="submitted",kernel=ABLATION,
               version=version,main_completed=9)
        return
    info=json.loads(launch.read_text(encoding="utf-8"))
    version=int(info["version"])
    ab_status=check(version,ABLATION)
    if ab_status in ("RUNNING","QUEUED","PENDING"):
        update(stage="ablation",status=ab_status,version=version,main_completed=9)
        return
    if ab_status not in ("COMPLETE","ERROR","CANCELLED"):
        update(stage="ablation",status=ab_status,version=version,main_completed=9)
        return
    abroot=download(version,ABLATION)
    ab=collect(abroot)
    expected={(m,s) for s in (42,123,2025) for m in ("ablation_axis","ablation_local")}
    missing_ab=sorted(expected-ab)
    if missing_ab:
        update(stage="ablation",status="incomplete",version=version,main_completed=9,
               ablation_completed=len(expected)-len(missing_ab),missing=missing_ab)
        return
    verify_manifest(abroot)
    update(stage="complete",status="all_15_runs_saved_locally",main_completed=9,
           ablation_completed=6,version=version,manuscript_alignment="requires pHash correction")
    # Do not repeatedly poll after a full and verified study.
    if os.name=="nt":
        subprocess.run(["schtasks","/Change","/TN","SALF_DINO_Kaggle_Result_Sync",
                        "/Disable"],capture_output=True,text=True,timeout=20)
if __name__=="__main__":
    try:main()
    except Exception as e:
        update(stage="monitor",status="error",error_type=type(e).__name__,message=str(e)[:500])
        raise
