import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHTMEM_SRC_DIR = os.path.join(CURRENT_DIR, "src")
if LIGHTMEM_SRC_DIR not in sys.path:
    sys.path.insert(0, LIGHTMEM_SRC_DIR)

from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config
from lightmem.factory.pre_compressor.llmlingua_2 import LlmLingua2Compressor
from lightmem.factory.topic_segmenter.llmlingua_2 import LlmLingua2Segmenter


def create_server(host, port, model_name, device, compression_rate):
    compressor_config = LlmLingua2Config(
        llmlingua_config={
            "model_name": model_name,
            "device_map": device,
            "use_llmlingua2": True,
        },
        llmlingua2_config={"max_batch_size": 50, "max_force_token": 100},
        compress_config={"instruction": "", "rate": compression_rate, "target_token": -1},
    )
    compressor = LlmLingua2Compressor(compressor_config)
    segmenter = LlmLingua2Segmenter(
        config={"model_name": model_name, "device_map": device},
        shared=True,
        compressor=compressor,
    )
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {"status": "ok", "model": model_name, "device": device})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                with inference_lock:
                    if self.path == "/compress":
                        messages = compressor.compress(payload.get("messages", []), segmenter.tokenizer)
                        self._write_json(200, {"messages": messages})
                        return
                    if self.path == "/propose_cut":
                        boundaries = segmenter.propose_cut(payload.get("buffer_texts", []))
                        self._write_json(200, {"boundaries": boundaries})
                        return
                self._write_json(404, {"error": "not found"})
            except Exception as exc:
                print(f"LLMLingua service error on {self.path}: {type(exc).__name__}: {exc}", flush=True)
                self._write_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format_string, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compression-rate", type=float, default=0.6)
    args = parser.parse_args()

    server = create_server(args.host, args.port, args.model, args.device, args.compression_rate)
    print(f"LLMLingua service ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
