import json
import os
import urllib.request
from pathlib import Path

url = "http://127.0.0.1:8000/analyze?stream=true"
boundary = "----WebKitFormBoundaryCheck"
sample_dir = Path(__file__).resolve().parent.parent / "sample_ecg"
if not sample_dir.exists():
    sample_dir = Path(__file__).resolve().parent / "sample_ecg"
file_path = sample_dir / "a01.dat"

with open(file_path, "rb") as f:
    file_bytes = f.read()

body = bytearray()
body.extend(f"--{boundary}\r\n".encode("ascii"))
body.extend(f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(file_path)}"\r\n'.encode("ascii"))
body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
body.extend(file_bytes)
body.extend(b"\r\n")
body.extend(f"--{boundary}\r\n".encode("ascii"))
body.extend(b'Content-Disposition: form-data; name="gestational_age"\r\n\r\n30.0\r\n')
body.extend(f"--{boundary}--\r\n".encode("ascii"))

req = urllib.request.Request(
    url,
    data=bytes(body),
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "text/event-stream"
    }
)

final_result = None
with urllib.request.urlopen(req) as resp:
    for line in resp:
        text = line.decode("utf-8").strip()
        if text.startswith("data:"):
            payload = json.loads(text.replace("data:", "").strip())
            if "result" in payload:
                final_result = payload["result"]

if final_result:
    print("SUCCESS: Received final result from stream!")
    print("Has signals:", "signals" in final_result)
    if "signals" in final_result:
        sig = final_result["signals"]
        print("Maternal points:", len(sig["maternal"]))
        print("Fetal points:", len(sig["fetal"]))
        print("IMF strips:", len(sig["imfs"]))
    print("Has biomarkers:", len(final_result.get("biomarkers", [])))
    print("Cardiac age:", final_result.get("cardiac_age"))
else:
    print("Failed to get final result")
