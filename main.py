import uvicorn
from typing import Union
from fastapi import FastAPI
from models import senA

app = FastAPI()


@app.get('/')
def read_root():
    return {"Hello World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get('/sentiment-analysis')
def SentimentAnalysis(jsondata=None):
    print('this is sentiment')
    return {jsondata}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
