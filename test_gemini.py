from models.google_gemini import generate_text, generate_json
from pydantic import BaseModel

# Simple text test
print("=== TEXT TEST ===")
print(generate_text("Say 'hello' in one short sentence."))

# Simple JSON test
class SimpleResult(BaseModel):
    message: str

print("\n=== JSON TEST ===")
res = generate_json(
    "Respond with a JSON object with a single field 'message' saying hello.",
    SimpleResult,
)
print(res)
