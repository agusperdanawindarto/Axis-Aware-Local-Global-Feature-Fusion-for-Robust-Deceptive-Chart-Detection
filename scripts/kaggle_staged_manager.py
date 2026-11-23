#!/usr/bin/env python3
"""Resume SALF-DINO in bounded Kaggle sessions. Run only when Windows is on.
Never report a completed experiment without validation of completion markers.
"""
from pathlib import Path
import subprocess,sys,os,json,hashlib,re,shutil,zipfile,datetime
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"KAGGLE"
STORAGE=ROOT/"SALF-DINO-GitHub"/"local_artifacts"
STORAGE.mkdir(parents=True,exist_ok=True)
CHECKPOINTS=BASE/"private_resume_dataset"
UP=BASE/"upload"
KERNEL="agusperdana/notebook38f0f2724f"
EXPECTED_SPLIT="c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a"
EPOCHS=20
STAGE_SIZE=4
PLAN=([(m,s) for s in (42,123,2025) for m in
      ("dinov3_global_avg","proposed_salf","swinv2_tiny")]
      +[(m,s) for s in (42,123,2025) for m in
      ("ablation_axis","ablation_local")])
STATE=Path(__file__).with_name("continuation_state.json")
LOG=Path(__file__).with_name("continuation_events.jsonl")

def log(event,**values):
    entry={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "event":event,**values}
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as f:f.write(json.dumps(entry,default=str)+"\n")
    print(json.dumps(entry,ensure_ascii=False,default=str),flush=True)
def readstate():
    if STATE.exists():return json.loads(STATE.read_text(encoding="utf-8"))
    return dict(kernel=KERNEL,version=8,model="dinov3_global_avg",seed=42,
                last_known_epoch=8,phase="running",full_plan=PLAN,
                started_from_version=7)
def savestate(st):
    temp=STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(st,indent=2),encoding="utf-8")
    os.replace(temp,STATE)
def env():
    e=os.environ.copy()
    for var,relative in {"TEMP":"cache/tmp","TMP":"cache/tmp",
      "PIP_CACHE_DIR":"cache/pip","HF_HOME":"cache/huggingface"}.items():
        path=ROOT/relative
        path.mkdir(parents=True,exist_ok=True)
        e[var]=str(path)
    return e
def kaggle(args,timeout=120):
    exe=BASE/".venv"/"Scripts"/"kaggle.exe"
    q=subprocess.run([str(exe)]+args,capture_output=True,text=True,
                     env=env(),timeout=timeout)
    if q.returncode!=0:
        raise RuntimeError(f'Kaggle CLI failed {args}: {q.stderr[-600:]}')
    return q.stdout
def status(v):
    s=kaggle(["kernels","status",f"{KERNEL}/{v}"],35)
    m=re.search(r"KernelWorkerStatus[.]([A-Z_]+)",s)
    return m.group(1) if m else "UNKNOWN"
def completed_record_path(folder,model,seed):
    return folder/f"paper_{model}_{seed}_completed.json"
def getcurrent_checkpoint(model,seed):
    return CHECKPOINTS/f"paper_{model}_{seed}_last.pt"
def epoch_info(model,seed):
    file=getcurrent_checkpoint(model,seed)
    if not file.exists():return (0,None)
    if not zipfile.is_zipfile(file):raise ValueError("Invalid Torch zip checkpoint "+str(file))
    with zipfile.ZipFile(file) as z:
        if z.testzip():raise ValueError("Checkpoint ZIP integrity failure")
    state=torch.load(file,map_location="cpu",weights_only=True)
    cfg=state["config"]
    assert cfg["mode"]=="paper" and cfg["epochs"]==20
    assert cfg["name"]==model and cfg["seed"]==seed
    assert cfg["split"]==EXPECTED_SPLIT
    assert cfg["dino"]=="vit_large_patch16_dinov3.lvd1689m"
    return int(state["epoch"]),state
def file_hash(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()
def download(v):
    dst=STORAGE/f"kaggle_v{v}"
    dst.mkdir(parents=True,exist_ok=True)
    pattern=r"(paper_.*|split_manifest[.]csv|label_audit[.]csv|checkpoint_audit[.]json|resumed_stage_.*[.]json)"
    text=kaggle(["kernels","output",f"{KERNEL}/{v}",
       "-p",str(dst),"--page-size","200","--file-pattern",pattern],180)
    files=[x for x in dst.rglob("*") if x.is_file() and x.suffix in (".pt",".csv",".json")]
    log("download_attempt",version=v,files=len(files),directory=str(dst),last=text[-500:])
    if not files:raise RuntimeError("No saved checkpoint/results returned by Kaggle version "+str(v))
    return dst
def update_checkpoint_dataset(archive,v):
    files=list(archive.rglob("paper_*_last.pt"))+list(archive.rglob("paper_*_best.pt"))
    for f in files:
        shutil.copy2(f,CHECKPOINTS/f.name)
    needed=[CHECKPOINTS/x.name for x in files]
    assert needed
    for f in needed:
        if not zipfile.is_zipfile(f):raise ValueError("Checkpoint file not a valid zip "+f.name)
        with zipfile.ZipFile(f) as z:
            if z.testzip() is not None:raise ValueError("Corrupt checkpoint "+f.name)
    manifest={
      "split_sha256":EXPECTED_SPLIT,
      "source_kaggle_version":v,
      "checkpoints":{f.name:file_hash(f) for f in CHECKPOINTS.glob("paper_*_*.pt")}
    }
    (CHECKPOINTS/"checkpoint_audit.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if files:
        notes=f"Updated validated checkpoints from Kaggle Version {v}"
        result=kaggle(["datasets","version","-p",str(CHECKPOINTS),
                       "-m",notes,"-q"],120)
        if "version error" in result.lower() or "error:" in result.lower():
            raise RuntimeError("Private checkpoint update rejected: "+result[-700:])
        log("private_checkpoint_dataset_updated",kaggle_version=v,
            files=len(manifest["checkpoints"]),result=result[-400:])
    return manifest
def inspect_completed(archive,model,seed):
    matches=list(archive.rglob(f"paper_{model}_{seed}_completed.json"))
    if not matches:return None
    if len(matches)!=1:raise RuntimeError("Ambiguous completion record")
    item=json.loads(matches[0].read_text(encoding="utf-8"))
    cfg=item["config"]
    assert cfg["name"]==model and cfg["seed"]==seed
    assert cfg["split"]==EXPECTED_SPLIT
    metrics=item["metrics"]
    assert metrics["model"]==model and int(metrics["seed"])==seed
    assert 0<=float(metrics["macro_f1"])<=1
    for suffix in ("predictions.csv","history.csv","cm.csv","by_category.csv","corruptions.csv"):
        required=list(archive.rglob(f"paper_{model}_{seed}_{suffix}"))
        if not required:raise RuntimeError("Missing evaluation file "+suffix)
    return metrics
def next_in_plan(model,seed):
    idx=PLAN.index((model,seed))
    return PLAN[idx+1] if idx+1<len(PLAN) else None
def make_next_notebook(model,seed,epoch_before):
    source=BASE/"Kaggle_SALF_DINO_Resume_From_Epoch_8.ipynb"
    assert source.exists()
    nb=json.loads(source.read_text(encoding="utf-8"))
    def find(t):
        items=[i for i,c in enumerate(nb["cells"]) if t in "".join(c["source"])]
        if len(items)!=1:raise RuntimeError(f"Notebook cell for {t} ambiguous: {items}")
        return items[0]
    def setcell(idx,s):nb["cells"][idx]["source"]=s.splitlines(keepends=True)
    restoreidx=find("PREVIOUS_VERSION_ROOT=None")
    restore="".join(nb["cells"][restoreidx]["source"])
    pos=restore.index("# Import verified private checkpoint from the old Kaggle run.")
    restore=restore[:pos]
    generic=r"""
# Import all verified private model checkpoints.
candidate=[f for f in Path('/kaggle/input').rglob('checkpoint_audit.json')
          if f.parent.name=='salf-dino-private-epoch-checkpoints']
assert len(candidate)==1,('Missing private Kaggle checkpoint source',candidate)
audit=json.loads(candidate[0].read_text(encoding='utf-8'))
assert audit['split_sha256']=='c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a'
for fname,digest in audit['checkpoints'].items():
    source=candidate[0].parent/fname
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    dest=ROOT/'checkpoints'/fname
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(source,dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
print('IMPORTED PRIVATE CHECKPOINTS',len(audit['checkpoints']),flush=True)
"""
    setcell(restoreidx,restore+generic)
    cfgidx=find("MAX_EPOCHS_THIS_VERSION = 4")
    cfg="".join(nb["cells"][cfgidx]["source"])
    assert "MODE = 'paper'" in cfg and "RUN_TRAINING = True" in cfg
    cfg=cfg.replace("MAX_EPOCHS_THIS_VERSION = 4  # resume from epoch 8, stop after epoch 12",
        "MAX_EPOCHS_THIS_VERSION = 4  # bounded four-epoch experiment session")
    if model in ("ablation_axis","ablation_local"):
        assert "RUN_ABLATIONS = False" in cfg
        cfg=cfg.replace("RUN_ABLATIONS = False","RUN_ABLATIONS = True",1)
    setcell(cfgidx,cfg)
    rid=find("# Resume a bounded four-epoch stage.")
    run=f"""# Single model/seed controlled continuation under the original 20-epoch schedule.
assert MODE=="paper" and EPOCHS==20 and MAX_EPOCHS_THIS_VERSION==4
assert torch.cuda.is_available()
expected_manifest="c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a"
assert hashlib.sha256((DATA_DIR/"split_manifest.csv").read_bytes()).hexdigest()==expected_manifest
model={model!r}
seed={seed}
before={epoch_before}
file=ROOT/"checkpoints"/f"paper_{{model}}_{{seed}}_last.pt"
if before:
    assert file.exists(),("Missing expected resume checkpoint",file)
    previous=torch.load(file,map_location="cpu",weights_only=True)
    assert int(previous["epoch"])==before,("Unexpected resume epoch",previous["epoch"],before)
else:
    assert not file.exists(),"No previous checkpoint expected for new model-seed run"
print("STAGED MODEL START",model,"SEED",seed,"PREVIOUS EPOCH",before,flush=True)
result=do_experiment(model,seed)
assert result.get("model")==model and int(result.get("seed"))==seed
if result.get("status")=="partial":
    assert before<int(result["epochs_completed"])<=min(20,before+MAX_EPOCHS_THIS_VERSION)
    print("STAGE PARTIAL CHECKPOINT VERIFIED",model,seed,result["epochs_completed"],flush=True)
else:
    assert result.get("macro_f1") is not None and 0<=result["macro_f1"]<=1
    print("COMPLETE TEST RESULTS",model,seed,result["macro_f1"],flush=True)
(RESULT_DIR/f"stage_result_{{model}}_{{seed}}.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8")
"""
    setcell(rid,run)
    nb["cells"][0]["source"]=(f"# SALF-DINO controlled stage: {model}, seed {seed}\n\n"
      f"Previous saved epoch: {epoch_before}. Up to four epochs in this Kaggle session, "
      "with original 20-epoch LR schedule and unchanged split. Checkpoint source is private. "
      "Resume is contingent upon verifying the last checkpoint, model settings and SHA-256.\n").splitlines(keepends=True)
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"stage_cell_{i}","exec")
    p=UP/"notebook38f0f2724f.ipynb"
    p.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
    return p
def push(model,seed,epoch_before):
    make_next_notebook(model,seed,epoch_before)
    meta=json.loads((UP/"kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"]==KERNEL
    assert meta["is_private"] and meta["enable_gpu"]
    assert "agusperdana/salf-dino-private-epoch-checkpoints" in meta["dataset_sources"]
    result=kaggle(["kernels","push","-p",str(UP),
                   "--accelerator","NvidiaTeslaT4","-t","7200"],120)
    m=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",result)
    if not m:raise RuntimeError("Notebook submission not confirmed: "+result[-700:])
    v=int(m.group(1))
    log("stage_submitted",version=v,model=model,seed=seed,previous_epoch=epoch_before)
    return v
def main():
    state=readstate()
    if state["phase"]=="all_study_finished":
        log("study_complete_no_action")
        return
    if state["phase"] in ("blocked","submitting"):
        log("manual_review_required",phase=state["phase"],version=state["version"])
        return
    v=int(state["version"])
    current_status=status(v)
    log("check",version=v,status=current_status,
        model=state["model"],seed=state["seed"])
    if current_status in ("RUNNING","QUEUED","PENDING","PREPARING"):
        savestate(state);return
    if current_status not in ("COMPLETE","ERROR","CANCEL_ACKNOWLEDGED","CANCELLED"):
        savestate(state);return
    archive=download(v)
    previous=int(state["last_known_epoch"])
    model=state["model"]
    seed=int(state["seed"])
    new_epoch,head_state=epoch_info(model,seed)
    local_files=list(archive.rglob(f"paper_{model}_{seed}_last.pt"))
    if local_files:
        uploaded=torch.load(local_files[0],map_location="cpu",weights_only=True)
        assert uploaded["config"]["name"]==model and int(uploaded["config"]["seed"])==seed
        assert uploaded["config"]["split"]==EXPECTED_SPLIT
        assert int(uploaded["epoch"])>=previous
        new_epoch=int(uploaded["epoch"])
    result=inspect_completed(archive,model,seed)
    if result is None and new_epoch<=previous:
        state.update(phase="blocked",error="No valid new checkpoint / no completed run",server_status=current_status)
        savestate(state)
        log("blocked",reason=state["error"],version=v)
        return
    update_checkpoint_dataset(archive,v)
    if result is not None:
        log("model_seed_complete",model=model,seed=seed,test_macro_f1=result["macro_f1"],version=v)
        nxt=next_in_plan(model,seed)
        if nxt is None:
            state.update(phase="all_study_finished",last_completed_version=v,last_test_macro_f1=result["macro_f1"])
            savestate(state);log("all_fifteen_model_seed_runs_complete")
            return
        model,seed=nxt
        before=0
    else:
        before=new_epoch
        log("checkpoint_continuation",model=model,seed=seed,epoch=new_epoch)
    # Persist a transactional intent before submitting; never duplicate a push on retry.
    state.update(phase="submitting",pending_model=model,pending_seed=seed,
                 pending_epoch=before,last_completed_version=v)
    savestate(state)
    new_v=push(model,seed,before)
    state.update(version=new_v,model=model,seed=seed,last_known_epoch=before,
                 phase="running",last_completed_version=v)
    for key in ("pending_model","pending_seed","pending_epoch"):
        state.pop(key,None)
    savestate(state)
if __name__=="__main__":
    try:main()
    except Exception as err:
        log("manager_error",type=type(err).__name__,error=str(err)[:650])
        sys.exit(1)
