"""FastAPI backend: một server giữ cả 4 mô hình."""
import base64, importlib, io, json, logging, sys, time
from contextlib import asynccontextmanager
from pathlib import Path
from starlette.concurrency import run_in_threadpool
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image,UnidentifiedImageError
from pydantic import BaseModel,Field
from config import CORS_ORIGINS,DEVICE,ENABLED_MODELS,MAX_UPLOAD_MB,ROOT,resolve_path
MODELS={}; LOAD_ERRORS={}; LOADERS={"classifier":("core.classifier","ImageClassifier"),"detector":("core.detector","ObjectDetector"),"retrieval":("core.retrieval","ImageSearch"),"llm":("core.llm","RAGChatbot")}
def _load_models():
    for name,(module,cls) in LOADERS.items():
        if name in ENABLED_MODELS:
            try: MODELS[name]=getattr(importlib.import_module(module),cls)()
            except Exception as exc: LOAD_ERRORS[name]=str(exc)
@asynccontextmanager
async def lifespan(app): _load_models(); yield; MODELS.clear()
app=FastAPI(title="AI Web Apps API",version="1.0.0",lifespan=lifespan); app.add_middleware(CORSMiddleware,allow_origins=CORS_ORIGINS,allow_methods=["*"],allow_headers=["*"])
def _require(name):
    if name not in MODELS: raise HTTPException(503,f"Mô hình '{name}' chưa được nạp")
    return MODELS[name]
async def _read_image(file):
    data=await file.read(MAX_UPLOAD_MB*1024*1024+1)
    if len(data)>MAX_UPLOAD_MB*1024*1024: raise HTTPException(413,"Ảnh quá lớn")
    try: im=Image.open(io.BytesIO(data)); im.load(); return im.convert("RGB")
    except (UnidentifiedImageError,OSError): raise HTTPException(400,"File không phải ảnh hợp lệ")
def _to_base64(image):
    b=io.BytesIO(); image.save(b,format="JPEG",quality=88); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
@app.get("/api/health")
def health(): return {"status":"ok","device":DEVICE,"models":{m:m in MODELS for m in sorted(ENABLED_MODELS)},"errors":LOAD_ERRORS}
@app.post("/api/classify")
async def classify(file:UploadFile=File(...),top_k:int=Form(3)):
    t=time.perf_counter(); result=await run_in_threadpool(_require("classifier").predict,await _read_image(file),max(1,min(top_k,5))); return {**result,"latency_ms":round((time.perf_counter()-t)*1000,1)}
@app.post("/api/detect")
async def detect(file:UploadFile=File(...),conf:float=Form(0.25)):
    t=time.perf_counter(); result,annotated=await run_in_threadpool(_require("detector").detect,await _read_image(file),conf=min(max(conf,.05),.95)); return {**result,"image":_to_base64(annotated),"latency_ms":round((time.perf_counter()-t)*1000,1)}
class TextQuery(BaseModel): query:str=Field(...,min_length=1,max_length=200); k:int=Field(8,ge=1,le=24)
def _urls(rs): return [{k:v for k,v in r.items() if k!="path"}|{"url":f"/api/gallery/{r['id']}"} for r in rs]
@app.post("/api/search/text")
def search_text(q:TextQuery): return {"results":_urls(_require("retrieval").search_text(q.query,q.k))}
@app.post("/api/search/image")
async def search_image(file:UploadFile=File(...),k:int=Form(8)): return {"results":_urls(await run_in_threadpool(_require("retrieval").search_image,await _read_image(file),max(1,min(k,24))))}
@app.get("/api/gallery/{item_id}")
def gallery(item_id:int):
    e=_require("retrieval"); return FileResponse(resolve_path(e.meta[item_id]["path"]))
class ChatRequest(BaseModel): message:str=Field(...,min_length=1,max_length=1000); history:list[dict]=Field(default_factory=list)
@app.post("/api/chat")
def chat(req:ChatRequest):
    contexts,tokens=_require("llm").stream(req.message,req.history)
    def events():
        yield f"data: {json.dumps({'type':'sources','items':contexts},ensure_ascii=False)}\n\n"
        for p in tokens: yield f"data: {json.dumps({'type':'token','text':p},ensure_ascii=False)}\n\n"
        yield 'data: {"type":"done"}\n\n'
    return StreamingResponse(events(),media_type="text/event-stream; charset=utf-8")
@app.post("/api/chat/sync")
def chat_sync(req:ChatRequest): return _require("llm").answer(req.message,req.history)
DIST=ROOT/"web"/"dist"
if DIST.exists(): app.mount("/",StaticFiles(directory=DIST,html=True),name="web")
