from pathlib import Path
import os,sys,subprocess,json,hashlib,datetime
R=Path(r"D:\RISETKOLABORASI\EPHI\PROJECT ke-2 IEEE\EXPERIMEN")
state=dict(time=datetime.datetime.now().isoformat(),project=str(R),checks={})
def record(name,success,info):
    state["checks"][name]=dict(ok=bool(success),detail=str(info)[:350])
    print(("PASS" if success else "BLOCK"),name,str(info)[:250],flush=True)
def sha(file):
    h=hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
expected="c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a"
archive=R/"dataset"/"DVCS_LOCAL_VERIFIED.json"
if archive.exists():
    data=json.loads(archive.read_text(encoding="utf-8"))
    record("dataset_extracted",data.get("image_count")==24667,data.get("image_count"))
else:record("dataset_extracted",False,"No verified local DVCS extraction")
manifest=R/"dataset"/"split_manifest.csv"
record("split_manifest",manifest.exists() and sha(manifest)==expected if manifest.exists() else False,
       sha(manifest) if manifest.exists() else "not ready")
checkpoint=R/"checkpoints"/"paper_dinov3_global_avg_42_last.pt"
best=R/"checkpoints"/"paper_dinov3_global_avg_42_best.pt"
torch=None
try:
    import torch
    cuda_ready=torch.cuda.is_available()
    record("cuda_torch",cuda_ready,torch.__version__+" CUDA build="+str(torch.version.cuda))
except Exception as e:record("cuda_torch",False,str(e))
if torch is not None and checkpoint.is_file() and best.is_file():
    try:
        a=torch.load(checkpoint,map_location="cpu",weights_only=True)
        b=torch.load(best,map_location="cpu",weights_only=True)
        c=a["config"]
        good=(c["split"]==expected and c["mode"]=="paper"
              and c["epochs"]==20 and c["name"]=="dinov3_global_avg"
              and c["seed"]==42 and c["res"]==256
              and b["signature"]==a["signature"] and b["epoch"]<=a["epoch"])
        record("recoverable_checkpoint",good,
              f"latest epoch={a['epoch']}, best validation macro-F1={b['best_f1']}")
    except Exception as e:record("recoverable_checkpoint",False,repr(e))
else:record("recoverable_checkpoint",False,"Missing paired checkpoint")
try:
    z=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                       "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
    c,mem,util=[int(v.strip()) for v in z.stdout.splitlines()[0].split(",")]
    record("gpu_safe_idle",c<=78 and mem<=2048,f"temp={c} C, used={mem} MiB, util={util}%")
except Exception as e:record("gpu_safe_idle",False,str(e))
# Running Kaggle job should finish before attempting local GPU continuation.
try:
    kaggle=R/"KAGGLE"/".venv"/"Scripts"/"kaggle.exe"
    z=subprocess.run([str(kaggle),"kernels","status",
       "agusperdana/notebook38f0f2724f/8"],capture_output=True,text=True,timeout=30)
    status=z.stdout.strip()
    running=("RUNNING" in status or "QUEUED" in status)
    record("cloud_session_finished",not running and z.returncode==0,status)
except Exception as e:record("cloud_session_finished",False,str(e))
weights=R/"cache"/"huggingface"/"PRETRAINED_BACKBONE_MANIFEST.json"
if weights.exists():
    known=json.loads(weights.read_text(encoding="utf-8"))
    ok=all(v.get("status")=="verified" for v in known.values()) and len(known)==2
    record("pretrained_weights_cached",ok,[(k,v.get("status")) for k,v in known.items()])
else:record("pretrained_weights_cached",False,"Pretrained weights not yet cached")
state["READY_FOR_GPU_TRAINING"]=all(v["ok"] for v in state["checks"].values())
out=R/"LOCAL_GPU_RESUME_PREFLIGHT.json"
out.write_text(json.dumps(state,indent=2),encoding="utf-8")
print("READY_FOR_GPU_TRAINING",state["READY_FOR_GPU_TRAINING"],flush=True)
