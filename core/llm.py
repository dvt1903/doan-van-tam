"""Ứng dụng 4 — Chatbot RAG: embeddings + FAISS + Qwen."""
import re, threading
from pathlib import Path
import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from config import DATA_DIR, DEVICE, EMBED_MODEL, LLM_MODEL
SYSTEM_PROMPT="Bạn là trợ lý ShopLite. Chỉ trả lời dựa trên TÀI LIỆU. Nếu không có thông tin, hãy nói chưa có thông tin và liên hệ hotline 1900 0000. Trả lời tiếng Việt và ghi nguồn [tên_file]."
def load_chunks(kb_dir:Path=DATA_DIR/"kb",max_chars:int=600):
    chunks=[]
    for path in sorted(Path(kb_dir).glob("*.md")):
        for section in re.split(r"\n(?=## )",path.read_text(encoding="utf-8")):
            section=section.strip()
            if section: chunks.append({"source":path.name,"text":section[:max_chars]})
    return chunks
class Retriever:
    def __init__(self,chunks,model_name=EMBED_MODEL):
        self.chunks=chunks; self.embedder=SentenceTransformer(model_name,device=DEVICE); embs=self.embedder.encode([c["text"] for c in chunks],normalize_embeddings=True,convert_to_numpy=True).astype("float32"); self.index=faiss.IndexFlatIP(embs.shape[1]); self.index.add(embs)
    def search(self,query,k=3):
        q=self.embedder.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32"); scores,ids=self.index.search(q,k); return [{**self.chunks[i],"score":round(float(s),4)} for s,i in zip(scores[0],ids[0]) if i!=-1]
class RAGChatbot:
    def __init__(self,kb_dir=DATA_DIR/"kb",model_name=LLM_MODEL):
        self.retriever=Retriever(load_chunks(kb_dir)); self.tokenizer=AutoTokenizer.from_pretrained(model_name); dtype=torch.float16 if DEVICE=="cuda" else torch.float32; self.model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype).to(DEVICE).eval(); self._lock=threading.Lock()
    def stream(self,question,history=None,k=3,max_new_tokens=180):
        contexts=self.retriever.search(question,k); docs="\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts); msgs=[{"role":"system","content":SYSTEM_PROMPT+"\n\nTÀI LIỆU:\n"+docs},{"role":"user","content":question}]; prompt=self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); inputs=self.tokenizer(prompt,return_tensors="pt").to(DEVICE); streamer=TextIteratorStreamer(self.tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=90); kwargs=dict(**inputs,streamer=streamer,max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.1)
        def it():
            with self._lock:
                t=threading.Thread(target=lambda:self.model.generate(**kwargs),daemon=True); t.start(); yield from streamer; t.join()
        return contexts,it()
    def answer(self,question,history=None,**kw):
        contexts,tokens=self.stream(question,history,**kw); return {"answer":"".join(tokens).strip(),"sources":contexts}
