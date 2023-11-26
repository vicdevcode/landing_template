from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import torch

device = torch.device("cpu")
np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("models/essays")

model = GPT2LMHeadModel.from_pretrained("models/essays")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class Text(BaseModel):
    text: str

@app.post("/generator")
async def generate(text: Text):
    text = text.text
    inpt = tok.encode(text, return_tensors="pt")
    out = model.generate(inpt.cpu(), max_length=500, repetition_penalty=5.0, do_sample=True, top_k=5, top_p=0.95, temperature=1)
    return {"generated": tok.decode(out[0])}
