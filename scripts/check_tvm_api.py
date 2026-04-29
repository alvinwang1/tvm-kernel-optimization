import modal

app = modal.App("tvm-check-api")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from modal_app import image
except ModuleNotFoundError:
    image = None

@app.function(image=image)
def check_api():
    from tvm import meta_schedule as ms
    db = ms.database.MemoryDatabase()
    print("Database methods:", dir(db))

@app.local_entrypoint()
def main():
    check_api.remote()
