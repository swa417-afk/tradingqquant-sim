from fastapi import FastAPI

app = FastAPI(title="TradingQ")


@app.get("/")
def read_root():
    return {"status": "ok"}
