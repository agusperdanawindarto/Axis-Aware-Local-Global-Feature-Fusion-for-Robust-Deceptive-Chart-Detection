#!/usr/bin/env python3
"""Download only experiment results/checkpoints from a completed Kaggle version.

Never downloads the original large DVCS dataset to Windows. All files and temp
caches are directed to the project on D:.
"""
from pathlib import Path
import subprocess,sys,os,shutil,re,argparse,json,hashlib
PROJECT=Path(__file__).resolve().parents[1]
MAIN_PROJECT=PROJECT.parent
DEFAULT_KERNEL="agusperdana/notebook38f0f2724f"
EXPECTED_MANIFEST="c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a"
def exe():
    opt=os.environ.get("KAGGLE_EXECUTABLE")
    if opt:return opt
    local=MAIN_PROJECT/"KAGGLE"/".venv"/"Scripts"/"kaggle.exe"
    if local.exists():return str(local)
    other=shutil.which("kaggle")
    if other:return other
    raise FileNotFoundError("Kaggle CLI not found. Install kaggle CLI or set KAGGLE_EXECUTABLE.")
def command(args,timeout=90):
    env=os.environ.copy()
    for key,relative in {"TEMP":"cache/tmp","TMP":"cache/tmp","TMPDIR":"cache/tmp",
        "PIP_CACHE_DIR":"cache/pip"}.items():
        path=MAIN_PROJECT/relative
        path.mkdir(parents=True,exist_ok=True)
        env[key]=str(path)
    result=subprocess.run([exe()]+args,capture_output=True,text=True,
        errors="replace",timeout=timeout,env=env)
    if result.returncode:
        raise RuntimeError("Kaggle command failed: "+result.stderr[-1200:])
    return result.stdout
def check(version,kernel):
    stdout=command(["kernels","status",kernel+"/"+str(version)],30)
    m=re.search(r"KernelWorkerStatus[.]([A-Z_]+)",stdout)
    return m.group(1) if m else stdout.strip()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--version",type=int,default=7)
    parser.add_argument("--kernel",default=DEFAULT_KERNEL)
    parser.add_argument("--force",action="store_true")
    a=parser.parse_args()
    state=check(a.version,a.kernel)
    print("KAGGLE_VERSION_STATUS",a.version,state,flush=True)
    if state in ("RUNNING","QUEUED","PENDING"):
        print("PENDING_NO_DOWNLOAD: Kaggle has not finalized this version.")
        return 0
    is_main=(a.kernel==DEFAULT_KERNEL)
    name=("kaggle_v"+str(a.version)) if is_main else ("kaggle_ablation_v"+str(a.version))
    target=PROJECT/"local_artifacts"/name
    target.mkdir(parents=True,exist_ok=True)
    marker=target/"_sync_complete.json"
    if marker.exists() and not a.force:
        print("ALREADY_COPIED",marker)
        return 0
    pattern=r"(paper_.*|split_manifest[.]csv|label_audit[.]csv|excluded_web_examples[.]txt)"
    args=["kernels","output",a.kernel+"/"+str(a.version),
          "-p",str(target),"--file-pattern",pattern]
    print("DOWNLOADING_ONLY_RESULTS_AND_CHECKPOINTS",target,flush=True)
    try:result=command(args,timeout=180)
    except Exception as e:
        print("DOWNLOAD_ERROR",str(e)[:350],flush=True)
        return 2
    print("OUTPUT",result[-800:],flush=True)
    files=[f for f in target.rglob("*") if f.is_file() and f!=marker]
    print("DOWNLOADED_FILES",len(files),flush=True)
    if not files:
        print("NO_EXPORTED_FILES_FOUND: will not mark sync complete",flush=True)
        return 3
    for f in files[:60]:print("FILE",f.relative_to(target),f.stat().st_size,flush=True)
    manifests=list(target.rglob("split_manifest.csv"))
    if manifests:
        digest=hashlib.sha256(manifests[0].read_bytes()).hexdigest()
        print("SPLIT_MANIFEST_SHA256",digest,flush=True)
        if digest!=EXPECTED_MANIFEST:
            print("WARNING: SPLIT MANIFEST DIFFERS FROM V6/V7 AUDIT",flush=True)
    done=list(target.rglob("paper_*_completed.json"))
    result=dict(kaggle_version=a.version,status=state,files=len(files),
          completed_model_seed_runs=len(done),
          all_nine_primary_runs_complete=len(done)>=9,
          split_manifest_present=bool(manifests))
    marker.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("SAVED_LOCAL_EXPORT",marker,flush=True)
    return 0
if __name__=="__main__":sys.exit(main())
