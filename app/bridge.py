import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from fastapi import FastAPI
from app.conf import utils as uts
from pydantic import BaseModel


app = FastAPI(title = 'fruver_price_forecast')

# Import Model
model = uts.load_artifact('model', 'artifacts/')

# Pydantic Input Requests Verif
class Inputs(BaseModel):
    product: str = ''
    market: str = ''

# Pydantic Output Verif
# class Outputs(BaseModel):

@app.post("/predict_product")
def post_predict_product(data: Inputs):
    response = model.predict_product(data.product)
    return response

@app.post("/predict_product_market")
def post_predict_product_market(data: Inputs):
    response = model.predict_product_market(data.product, data.market)
    return response


@app.get("/info")
def get_model_info():
    model_info = model.get_model_info()
    return model_info

if __name__ == '__main__':
    import uvicorn

    # For local development:
    uvicorn.run("bridge:app", port=3000, reload=True)

    # For Docker deployment:
    # uvicorn.run("bridge:app", host='0.0.0.0', port=80)