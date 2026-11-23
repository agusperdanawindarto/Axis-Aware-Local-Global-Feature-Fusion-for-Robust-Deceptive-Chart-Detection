# SALF-DINO fault-tolerant training engine. Executed from the notebook namespace.
import json, os, hashlib, random, time, gc
from pathlib import Path
AUTO_RESUME=True
EARLY_STOP_PATIENCE=5
EARLY_STOP_MIN_DELTA=0.0005
EARLY_STOP_MIN_EPOCHS=5
CHECKPOINT_DIR=ROOT/"checkpoints"
CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)

def create_optimizer(model):
    for p in model.encoder.parameters(): p.requires_grad_(False)
    head=[p for n,p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    assert head, "No trainable head"
    opt=torch.optim.AdamW(head,lr=5e-4,weight_decay=0.05)
    def mult(epoch):
        warm=min(2,EPOCHS)
        if epoch<warm: return (epoch+1)/max(1,warm)
        progress=(epoch+1-warm)/max(1,EPOCHS-warm)
        return 0.002+0.998*0.5*(1+math.cos(math.pi*min(1.0,progress)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,mult)

def epoch_train(model,loader,opt,scaler,amp_enabled):
    model.train();model.encoder.eval();opt.zero_grad(set_to_none=True)
    loss_total,n=0.,0
    for step,(x,y,_) in enumerate(tqdm(loader,leave=False)):
        x=x.to(DEVICE);y=y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,enabled=amp_enabled):
            logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=0.2)
        loss_total+=loss.item()*y.numel();n+=y.numel()
        tail=len(loader)%GRAD_ACCUM
        group=GRAD_ACCUM if step<len(loader)-(tail or GRAD_ACCUM) else (tail or GRAD_ACCUM)
        scaler.scale(loss/group).backward()
        if (step+1)%GRAD_ACCUM==0 or step+1==len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    return loss_total/max(n,1)

def _head(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith("encoder.")}

def _load_head(model,weights):
    expected={k for k in model.state_dict() if not k.startswith("encoder.")}
    assert set(weights)==expected,"Checkpoint keys don't match"
    missing=model.load_state_dict(weights,strict=False)
    assert not missing.unexpected_keys and all(k.startswith("encoder.") for k in missing.missing_keys)

def _save(obj,path):
    tmp=path.with_name(path.name+".tmp")
    try: torch.save(obj,tmp);os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

def _json(obj,path):
    tmp=path.with_name(path.name+".tmp")
    try:tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8");os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def _rng():
    npstate=np.random.get_state()
    return {"python":random.getstate(),
            "numpy":(str(npstate[0]),npstate[1].tolist(),int(npstate[2]),int(npstate[3]),float(npstate[4])),
            "torch":torch.get_rng_state().cpu(),
            "cuda":[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}

def _restore_rng(r):
    random.setstate(r["python"])
    q=r["numpy"]
    np.random.set_state((q[0],np.array(q[1],dtype=np.uint32),q[2],q[3],q[4]))
    torch.set_rng_state(r["torch"].cpu())
    if torch.cuda.is_available() and r["cuda"]:torch.cuda.set_rng_state_all(r["cuda"])

def _signature(name,seed,size):
    split=DATA_DIR/"split_manifest.csv"
    if not split.exists():raise FileNotFoundError(split)
    config=dict(name=name,seed=int(seed),mode=MODE,epochs=EPOCHS,res=size,
                batch=BATCH_SIZE,grad_accum=GRAD_ACCUM,dino=DINO_NAME,swin=SWIN_NAME,
                split=hashlib.sha256(split.read_bytes()).hexdigest(),repo=DATASET_ID,
                aug=USE_ROBUST_TRAIN_AUG,patience=EARLY_STOP_PATIENCE,
                min_delta=EARLY_STOP_MIN_DELTA,min_epochs=EARLY_STOP_MIN_EPOCHS,
                training_protocol="frozen-backbone-resume-v2")
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),config

def _matching(obj,sig,path):
    if obj.get("signature")!=sig:
        raise RuntimeError("Checkpoint/config mismatch at "+str(path)+". Archive old run before changing parameters.")

def do_experiment(name,seed):
    size=size_for(name);sig,config=_signature(name,seed,size)
    prefix=f"{MODE}_{name}_{seed}"
    last=CHECKPOINT_DIR/(prefix+"_last.pt")
    best=CHECKPOINT_DIR/(prefix+"_best.pt")
    marker=RESULT_DIR/(prefix+"_completed.json")
    required=[RESULT_DIR/(prefix+"_"+ext+".csv") for ext in ("predictions","history","cm","by_category","corruptions")]
    if marker.exists():
        done=json.loads(marker.read_text(encoding="utf-8"))
        _matching(done,sig,marker)
        if all(p.is_file() for p in required):
            print("SKIP completed:",prefix)
            return done["metrics"]
    if last.exists() and not AUTO_RESUME:raise RuntimeError("Existing last.pt; resume or archive it")
    if best.exists() and not last.exists():raise RuntimeError("Orphan best.pt; archive before new experiment")

    seed_everything(seed)
    model=build_model(name).to(DEVICE)
    optimizer,scheduler=create_optimizer(model)
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    train_loader,_=make_loader("train",size,train=True)
    val_loader,_=make_loader("val",size)
    test_loader,test_frame=make_loader("test",size)
    start_epoch=0;history=[];best_f1=-1.;best_epoch=0;wait=0;stopped=False
    if last.exists():
        state=torch.load(last,map_location="cpu",weights_only=True)
        _matching(state,sig,last)
        _load_head(model,state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.state.values():
            for k,v in group.items():
                if torch.is_tensor(v):group[k]=v.to(DEVICE)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"])
        start_epoch=state["epoch"];history=state["history"]
        best_f1=state["best_f1"];best_epoch=state["best_epoch"]
        wait=state["wait"];stopped=state["early_stopped"]
        assert start_epoch==len(history),"Saved history mismatch"
        print("RESUME",prefix,"NEXT EPOCH",start_epoch+1,"BEST",best_f1)
    for epoch in range(start_epoch,EPOCHS):
        if stopped:break
        seed_everything(seed+100003*(epoch+1))
        loss=epoch_train(model,train_loader,optimizer,scaler,DEVICE.type=="cuda")
        _,yt,yp,_,_=predict(model,val_loader)
        metrics=measure_metrics(yt,yp);score=float(metrics["macro_f1"])
        if not np.isfinite(score):raise RuntimeError("Nonfinite validation score")
        history.append(dict(epoch=int(epoch+1),loss=float(loss),**{k:float(v) for k,v in metrics.items()}))
        if best_epoch==0 or score>best_f1+EARLY_STOP_MIN_DELTA:
            best_f1=score;best_epoch=epoch+1;wait=0
            _save(dict(model=_head(model),signature=sig,epoch=best_epoch,best_f1=best_f1),best)
        else:wait+=1
        scheduler.step()
        stopped=epoch+1>=EARLY_STOP_MIN_EPOCHS and wait>=EARLY_STOP_PATIENCE
        _save(dict(signature=sig,config=config,model=_head(model),
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   scaler=scaler.state_dict(),rng=_rng(),epoch=epoch+1,
                   history=history,best_f1=best_f1,best_epoch=best_epoch,
                   wait=wait,early_stopped=stopped),last)
        pd.DataFrame(history).to_csv(RESULT_DIR/(prefix+"_history.csv"),index=False)
        print(prefix,"epoch",epoch+1,"val macro-F1",round(score,5),"best",round(best_f1,5),"patience",wait)
        if stopped:print("EARLY STOP",prefix,"best epoch",best_epoch);break
    if not best.exists():raise RuntimeError("Missing best.pt")
    winner=torch.load(best,map_location="cpu",weights_only=True)
    _matching(winner,sig,best)
    _load_head(model,winner["model"])
    idx,y,pred,prob,seconds=predict(model,test_loader)
    assert np.array_equal(idx,np.arange(len(test_frame)))
    per_image=test_frame.copy()
    per_image["pred"]=pred;per_image["prob_deceptive"]=prob
    per_image.to_csv(RESULT_DIR/(prefix+"_predictions.csv"),index=False)
    result=measure_metrics(y,pred)
    result.update(model=name,seed=int(seed),mode=MODE,epochs_completed=len(history),
                  best_epoch=best_epoch,early_stopped=bool(stopped),resolution=size,
                  params_m=sum(p.numel() for p in model.parameters())/1e6,
                  trainable_params_m=sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,
                  eval_images_per_second=len(y)/max(seconds,1e-9),val_best_macro_f1=best_f1)
    pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),
                 index=["true_0","true_1"],columns=["pred_0","pred_1"]).to_csv(
                 RESULT_DIR/(prefix+"_cm.csv"))
    cat=[]
    for c,group in per_image.groupby("category"):
        cat.append(dict(category=c,n=len(group),**measure_metrics(group.label.to_numpy(),group.pred.to_numpy())))
    pd.DataFrame(cat).to_csv(RESULT_DIR/(prefix+"_by_category.csv"),index=False)
    distortions=[]
    for kind in ["jpeg","blur","downsample"]:
        loader,_=make_loader("test",size,corruption=kind)
        _,yt,yp,_,_=predict(model,loader)
        distortions.append(dict(corruption=kind,**measure_metrics(yt,yp)))
    pd.DataFrame(distortions).to_csv(RESULT_DIR/(prefix+"_corruptions.csv"),index=False)
    pd.DataFrame(history).to_csv(RESULT_DIR/(prefix+"_history.csv"),index=False)
    _json(dict(signature=sig,config=config,metrics=result),marker)
    print("COMPLETED",prefix,result)
    del model,optimizer,scaler,train_loader,val_loader,test_loader
    gc.collect()
    if DEVICE.type=="cuda":torch.cuda.empty_cache()
    return result
