#!/usr/bin/env python3
"""Behavioral regressions for documentation reference and example validation."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_documentation_examples import validate_openapi
from test_http_regression import RegressionFailure


class DocumentationValidationTest(unittest.TestCase):
    def test_external_reference_is_rejected_without_http_request(self):
        requests = 0
        lock = threading.Lock()
        response = json.dumps({
            "description": "Example response",
            "content": {"application/json": {"schema": {"type": "string"}}},
        }).encode("utf-8")

        class ResponseHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal requests
                with lock:
                    requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), ResponseHandler)
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            reference = {
                "$ref": f"http://127.0.0.1:{server.server_port}/response.json",
            }
            for placement in ("path_response", "named_component_response"):
                with self.subTest(placement=placement):
                    document = {
                        "openapi": "3.0.1",
                        "info": {"title": "Reference regression", "version": "1.0.0"},
                        "paths": {"/example": {"get": {"responses": {"200": reference}}}},
                        "components": {"schemas": {"Literal": {
                            "type": "object",
                            "properties": {
                                "example": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["example", "type"],
                            "example": {"example": "literal", "type": "object"},
                        }}},
                    }
                    if placement == "named_component_response":
                        document["components"]["responses"] = {"example": reference}
                        document["paths"]["/example"]["get"]["responses"]["200"] = {
                            "$ref": "#/components/responses/example",
                        }
                    with self.assertRaises(RegressionFailure):
                        validate_openapi(document)
                    with lock:
                        self.assertEqual(requests, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        with lock:
            self.assertEqual(requests, 0)

    def test_literal_property_names_preserve_example_validation(self):
        document = {
            "openapi": "3.0.1",
            "info": {"title": "Literal regression", "version": "1.0.0"},
            "paths": {},
            "components": {"schemas": {"Literal": {
                "type": "object",
                "properties": {
                    "example": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["example", "type"],
                "example": {"example": "literal", "type": "object"},
            }}},
        }
        validate_openapi(document)
        del document["components"]["schemas"]["Literal"]["example"]["type"]
        with self.assertRaises(AssertionError):
            validate_openapi(document)


if __name__ == "__main__":
    unittest.main()
