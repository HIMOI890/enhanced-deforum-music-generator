from fastapi import FastAPI
<<<<<<< HEAD
from . import health, analysis, deforum, status
=======
from . import health, analysis, deforum, status, models
>>>>>>> 3595d08 (Initial import)

app = FastAPI(title="Enhanced Deforum Music Generator API")

app.include_router(health.router, prefix="/health")
app.include_router(analysis.router, prefix="/analysis")
app.include_router(deforum.router, prefix="/deforum")
app.include_router(status.router, prefix="/status")
<<<<<<< HEAD
=======
app.include_router(models.router, prefix="/models")
>>>>>>> 3595d08 (Initial import)
