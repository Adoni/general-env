from fastapi import FastAPI
import uvicorn
from http_service.modules.routes import router


app = FastAPI(title="算法评测工具")
app.include_router(router, prefix="/api/v1") 

uvicorn.run(app, host="0.0.0.0", port=10000)