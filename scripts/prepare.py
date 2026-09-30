"""Download public data, train ResNet-18 head, build CLIP index. No fake predictions."""
import sys,json,random,shutil,urllib.request,tarfile,zipfile,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from config import ROOT,DATA_DIR,ART_DIR,DEVICE
from core.classifier import build_model,EVAL_TF
def download(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        print('Download:',url,flush=True); tmp=dest.with_suffix(dest.suffix+'.part'); urllib.request.urlretrieve(url,tmp); tmp.replace(dest)
    return dest
def prepare(epochs=25):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    for name in ['classifier','detector','retrieval']:(ART_DIR/name).mkdir(parents=True,exist_ok=True)
    flowers=DATA_DIR/'flowers'/'flower_photos'
    if not flowers.exists():
        p=download('https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',DATA_DIR/'flower_photos.tgz'); tarfile.open(p).extractall(DATA_DIR/'flowers',filter='data')
    coco=DATA_DIR/'coco128'
    if not coco.exists():
        p=download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',DATA_DIR/'coco128.zip'); zipfile.ZipFile(p).extractall(DATA_DIR)
    if not (ART_DIR/'classifier/model.pt').exists():
        base=ImageFolder(flowers,transform=EVAL_TF); ids=np.arange(len(base)); train,remain=train_test_split(ids,test_size=.2,stratify=base.targets,random_state=42); val,test=train_test_split(remain,test_size=.5,stratify=np.array(base.targets)[remain],random_state=42)
        model=build_model(len(base.classes)).to(DEVICE); model.eval(); model.fc=torch.nn.Identity(); features=[];labels=[]
        with torch.inference_mode():
            for x,y in DataLoader(base,batch_size=32,num_workers=2):features.append(model(x.to(DEVICE)).cpu());labels.append(y)
        x=torch.cat(features).to(DEVICE);y=torch.cat(labels).to(DEVICE);model.fc=torch.nn.Linear(512,len(base.classes)).to(DEVICE);opt=torch.optim.AdamW(model.fc.parameters(),lr=.003);best=-1
        for _ in range(epochs):
            model.fc.train()
            for b in torch.randperm(len(train)).split(128):
                idx=torch.as_tensor(train[b.numpy()],device=DEVICE);opt.zero_grad();loss=torch.nn.functional.cross_entropy(model.fc(x[idx]),y[idx]);loss.backward();opt.step()
            model.fc.eval();score=(model.fc(x[val]).argmax(1)==y[val]).float().mean().item()
            if score>best:best=score;torch.save(model.state_dict(),ART_DIR/'classifier/model.pt')
        model.load_state_dict(torch.load(ART_DIR/'classifier/model.pt',weights_only=True,map_location=DEVICE));pred=model.fc(x[test]).argmax(1).cpu().numpy();truth=y[test].cpu().numpy();metrics={'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro'),'confusion_matrix':confusion_matrix(truth,pred).tolist(),'classes':base.classes,'test_size':len(test)};(ART_DIR/'classifier/classes.json').write_text(json.dumps(base.classes));(ART_DIR/'classifier/metrics.json').write_text(json.dumps(metrics,indent=2))
    download('https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt',ART_DIR/'detector/yolo11n.pt')
    gallery=DATA_DIR/'gallery';gallery.mkdir(exist_ok=True);items=[]
    for p in sorted((coco/'images/train2017').glob('*.jpg')):
        d=gallery/f'coco_{p.name}';shutil.copy2(p,d);items.append({'path':str(d.relative_to(ROOT)),'label':'COCO '+p.stem,'source':'coco'})
    rng=random.Random(42)
    for category in sorted(p for p in flowers.iterdir() if p.is_dir()):
        for p in rng.sample(sorted(category.glob('*.jpg')),min(100,len(list(category.glob('*.jpg'))))):
            d=gallery/f'{category.name}_{p.name}';shutil.copy2(p,d);items.append({'path':str(d.relative_to(ROOT)),'label':category.name,'source':'flowers'})
    if not (ART_DIR/'retrieval/index.faiss').exists():
        from core.retrieval import ClipEncoder,build_index;build_index(ClipEncoder(),items)
    from core.llm import RAGChatbot;RAGChatbot();print('All four models prepared.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=25);prepare(p.parse_args().epochs)
