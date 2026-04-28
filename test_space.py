import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from modal_app import image
except ModuleNotFoundError:
    image = None
import modal

app = modal.App("tvm-test-space")

@app.function(image=image, timeout=300)
def test_space():
    import tvm
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.space_generator import PostOrderApply
    from tvm.meta_schedule.schedule_rule import ScheduleRule
    
    # Try getting the default rules
    try:
        rules = ScheduleRule.create("cuda")
        filtered = [r for r in rules if "CrossThreadReduction" not in type(r).__name__]
        print(f"Success! Got {len(filtered)} filtered rules.")
    except Exception as e:
        print(f"Error: {e}")

@app.local_entrypoint()
def main():
    test_space.remote()
