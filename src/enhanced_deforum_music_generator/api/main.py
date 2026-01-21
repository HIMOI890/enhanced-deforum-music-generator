from fastapi import FastAPI
from . import health, analysis, deforum, status

app = FastAPI(title="Enhanced Deforum Music Generator API")

app.include_router(health.router, prefix="/health")
app.include_router(analysis.router, prefix="/analysis")
app.include_router(deforum.router, prefix="/deforum")
app.include_router(status.router, prefix="/status")
