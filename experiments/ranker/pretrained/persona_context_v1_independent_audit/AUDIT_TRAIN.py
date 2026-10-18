"""Archived-source CPU inference only; no optimizer/native calls."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import torch

BASE = Path('experiments/ranker/pretrained')
RUN = Path('data/ranker/pretrained/persona-context-v1-attempt-1')
OUT = BASE / 'persona_context_v1_independent_audit'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original = read(OUT / 'ORIGINAL_FILES_SHA256.json')['files']
assert all(sha(RUN / n) == h for n,h in original.items())
spec = importlib.util.spec_from_file_location('experiments.ranker.pretrained.persona_archived_audit', RUN / 'source.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.HERE = BASE.resolve()
torch.set_num_threads(1)
report=read(RUN/'results.json'); frozen=read(RUN/'ARCHITECTURE_AND_INPUTS_FROZEN.json')
assert sha(RUN/'source.py') == frozen['sourceSha256']
for p,h in frozen['dataSha256'].items(): assert sha(p)==h
for p,h in report['catalogPins'].items(): assert sha(p)==h
manifest=read('data/authored-history-tiger-v1/manifest.json')
for name,h in manifest['files'].items():
 p=Path('data/authored-history-tiger-v1')/name;assert sha(p)==h
 assert read(p)==read(BASE/'synthetic_history_v1'/name)
for name,h in manifest['originalSourceSha256'].items():assert sha(BASE/'synthetic_history_v1'/name)==h
profiles=read('data/authored-history-tiger-v1/profiles.public.json')['profiles']
train,held=m.checked_split(profiles,read(BASE/'synthetic_history_v1/split.json'))
catalog=m.load_catalog_features('/Users/lukas/Documents/Projects/dubhacks-2026/data/ranker/pretrained/actual-store-function-v3-attempt-1')
head,saved=m.load_head(RUN/'head.pt');ablation,asaved=m.load_head(RUN/'onehot-ablation.pt')
assert sha(RUN/'head.pt')==report['checkpointSha256']; assert sha(RUN/'onehot-ablation.pt')==report['onehotCheckpointSha256']
train_labels=m.label_rows(BASE/'synthetic_history_v1/labels.train.json')
text,history=m.source_inputs(train,catalog);colors,indices=m.source_colors(catalog)
pt,vt=m.targets(train,train_labels,catalog)
initial=m.PersonaContextHead()
def loss(model):
 with torch.no_grad():
  p,v=model(text,history,catalog['vectors'],colors,indices)
  return float(-(pt*p.log_softmax(-1)).sum(-1).mean()-m.CONFIG['variantLossWeight']*(vt*v.log_softmax(-1)).sum(-1).mean())
changed=[k for k,v in initial.state_dict().items() if not torch.equal(v,head.state_dict()[k])]
assert changed==report['weightChanges']['changedTensors']
outputs={'context-qwen-catalog':m.predict(head,held,catalog),'onehot-catalog-ablation':m.predict(ablation,held,catalog,catalog_vectors=torch.eye(16)),'no-history':m.predict(head,held,catalog,omit_history=True)}
for name,value in outputs.items():assert value==read(RUN/(name+'-predictions.json')),name
freeze=read(RUN/'PREDICTIONS_FROZEN.json')
for name,h in freeze['files'].items():assert sha(RUN/(name+'-predictions.json'))==h
heldlabels=m.label_rows(BASE/'synthetic_history_v1/labels.heldout.json')
family_by_group={g['userGroupId']:g['familyId'] for g in read(BASE/'synthetic_history_v1/split.json')['groups']}
for name in freeze['files']:
 value=m.evaluate_rows(read(RUN/(name+'-predictions.json')),heldlabels,catalog)
 families={}
 for row in value['rows']:families.setdefault(family_by_group[row['userGroupId']],[]).append(row)
 value['perFamily']={name:{'sessions':len(rows),'groups':len({r['userGroupId'] for r in rows}),'novelRelevantTop1Rate':sum(r['novelRelevantTop1'] for r in rows)/len(rows),'topGrade3Rate':sum(r['topGrade3'] for r in rows)/len(rows)} for name,rows in families.items()}
 assert value==report['heldoutMetrics'][name],name
pred=m.predict(head,train,catalog)
def metrics(predictions,labels):
 byid={l['profileId']:l for l in labels};products=[p['productId'] for p in catalog['products']];rows=[]
 for p in predictions:
  l=byid[p['profileId']];g=l['relevanceGrades'];top=products[p['order'][0]] if p['order'] else None
  gain=[2**g[products[i]]-1 for i in p['order'][:6]];ideal=sorted([2**v-1 for v in g.values()],reverse=True)[:6]
  dcg=lambda a:sum(v/math.log2(i+2) for i,v in enumerate(a))
  rows.append({'profileId':p['profileId'],'topProductId':top,'grade':g.get(top,0),'owned':top in l['alreadyOwnedProductIds'],'ndcg6':dcg(gain)/max(dcg(ideal),1e-8)})
 return {'cases':len(rows),'topGrade3':sum(r['grade']==3 for r in rows),'topGradeAtLeast2':sum(r['grade']>=2 for r in rows),'ownedTop1':sum(r['owned'] for r in rows),'ndcg6':sum(r['ndcg6'] for r in rows)/len(rows),'top1Histogram':dict(Counter(r['topProductId'] for r in rows)),'rows':rows}
progress=read(RUN/'OPTIMIZER_PROGRESS.json')
for n,h in original.items():assert sha(RUN/n)==h
trainout={'status':'pass-actual-training-and-replay; quality-not-promotion','optimizerExecutedByAuditor':False,'nativeExecutedByAuditor':False,'originalRunFilesUnchanged':len(original),'sourceSha256':sha(RUN/'source.py'),'initialJointTrainLoss':loss(initial),'finalJointTrainLoss':loss(head),'irreducibleSoftTargetEntropy':float(-(pt*pt.clamp_min(1e-12).log()).sum(-1).mean()-.5*(vt*vt.clamp_min(1e-12).log()).sum(-1).mean()),'train':metrics(pred,train_labels),'heldout':metrics(outputs['context-qwen-catalog'],heldlabels),'exactReplayedArms':list(outputs),'allSevenMetricsRecomputedExactly':True,'checkpointSha256':sha(RUN/'head.pt'),'actualOptimizerReceipt':progress,'changedTensors':changed,'tigerExportFiveDocumentsDeepEqualSeed':True,'predictionFreezeMtimeBeforeResults':(RUN/'PREDICTIONS_FROZEN.json').stat().st_mtime_ns<=(RUN/'results.json').stat().st_mtime_ns}
with (OUT/'TRAIN_DIAGNOSIS.json').open('x') as f:json.dump(trainout,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in trainout.items() if k not in ['train','heldout','actualOptimizerReceipt']}));print({split:{k:v for k,v in trainout[split].items() if k!='rows'} for split in ['train','heldout']})
