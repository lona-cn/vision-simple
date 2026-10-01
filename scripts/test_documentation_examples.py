#!/usr/bin/env python3
"""Validate published contracts and execute their unmodified inference examples."""

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import traceback
from urllib.parse import urlsplit

import yaml
from jsonschema import Draft202012Validator
from openapi_schema_validator import OAS30Validator
from openapi_spec_validator import validate
from PIL import Image
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from test_http_regression import Server, model_yaml, require
from test_protocol_regression import Session


DOCUMENT_URI = "https://vision-simple.invalid/openapi/server.yaml"


def walk(value, path=(), *, is_schema=False, map_kind=None, is_example=False):
    """Visit contract objects, distinguishing named maps from instance data."""
    if isinstance(value, dict):
        if map_kind is not None:
            for name, child in value.items():
                yield from walk(child, (*path, str(name)),
                                is_schema=map_kind == "schemas",
                                is_example=map_kind == "examples")
            return
        yield path, value, is_schema
        if is_schema:
            for name, child in value.get("properties", {}).items():
                yield from walk(child, (*path, "properties", str(name)), is_schema=True)
            for keyword in ("items", "additionalProperties", "not"):
                if isinstance(value.get(keyword), dict):
                    yield from walk(value[keyword], (*path, keyword), is_schema=True)
            for keyword in ("allOf", "anyOf", "oneOf"):
                for index, child in enumerate(value.get(keyword, [])):
                    yield from walk(child, (*path, keyword, str(index)), is_schema=True)
        else:
            for key, child in value.items():
                if key == "example" or (is_example and key == "value") or key.startswith("x-"):
                    continue
                child_path = (*path, str(key))
                named_map = key in ("paths", "responses", "headers", "content", "examples",
                                    "callbacks", "links", "encoding", "schemas", "parameters",
                                    "requestBodies", "securitySchemes") and isinstance(child, dict)
                yield from walk(child, child_path, is_schema=key == "schema",
                                map_kind=key if named_map else None)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, (*path, str(index)), is_schema=is_schema)


def location(path):
    return "#/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in path)


def schema_validator(schema, registry):
    return OAS30Validator(schema, registry=registry,
                          _resolver=registry.resolver(DOCUMENT_URI),
                          format_checker=OAS30Validator.FORMAT_CHECKER)


def validate_instance(schema, value, registry, context):
    try:
        schema_validator(schema, registry).validate(value)
    except Exception as error:
        raise AssertionError(f"{context}: {error}") from error


def validate_openapi(document):
    nodes = list(walk(document))
    # Reject external/file references before any standards validator can resolve
    # them. Literal example/default/enum data is not part of this contract walk.
    for path, node, _ in nodes:
        if "$ref" in node:
            reference = node["$ref"]
            require(isinstance(reference, str) and reference.startswith("#/"),
                    f"{location(path)}: expected a local documentation reference")
    validate(document)
    registry = Registry().with_resource(DOCUMENT_URI, Resource(document, DRAFT4))
    resolver = registry.resolver(DOCUMENT_URI)
    references = examples = 0
    for path, node, is_schema in nodes:
        if "$ref" in node:
            reference = node["$ref"]
            try:
                resolver.lookup(reference)
            except Exception as error:
                raise AssertionError(f"{location(path)}: unresolved {reference}") from error
            references += 1
        schema = node if is_schema else node.get("schema")
        if schema is not None and "example" in node:
            validate_instance(schema, node["example"], registry, location((*path, "example")))
            examples += 1
        if not is_schema and "schema" in node and "examples" in node:
            for name, example in node["examples"].items():
                if "$ref" in example:
                    example = resolver.lookup(example["$ref"]).contents
                if "value" in example:
                    validate_instance(node["schema"], example["value"], registry,
                                      location((*path, "examples", str(name), "value")))
                    examples += 1
    require(examples > 0, "OpenAPI contains no schema-bound examples")
    print(f"PASS OpenAPI standard validation, {references} local references, {examples} schema-bound examples", flush=True)
    return registry


def inference_examples(document):
    for kind in ("yolo", "ocr"):
        route = f"/v0/infer/{kind}"
        operation = document["paths"][route]["post"]
        media = operation["requestBody"]["content"]["application/json"]
        payload = media["example"]
        require(payload["images"], f"{route}: published sample needs an image")
        for index, encoded in enumerate(payload["images"]):
            context = f"{route} example images[{index}]"
            try:
                raw = base64.b64decode(encoded, validate=True)
                require(base64.b64encode(raw).decode("ascii") == encoded,
                        f"{context}: noncanonical base64")
                require(0 < len(raw) <= 16 * 1024, f"{context}: sample exceeds compact 16 KiB budget")
                with Image.open(io.BytesIO(raw)) as image:
                    require(0 < image.width <= 1024 and 0 < image.height <= 1024,
                            f"{context}: sample exceeds 1024x1024 pixel budget")
                    image.load()
                    print(f"PASS {context}: {len(raw)} bytes, {image.width}x{image.height} decoded {image.format}", flush=True)
            except Exception as error:
                raise AssertionError(f"{context}: {error}") from error
        response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        yield route, payload, response_schema


def mcp_template(root):
    config = json.loads((root / ".mcp.json.example").read_text(encoding="utf-8"))
    servers = config["mcpServers"]
    require(len(servers) == 1, "Published MCP template must identify one local server")
    entry = next(iter(servers.values()))
    require(set(entry) == {"type", "url"} and entry["type"] == "sse",
            "Published MCP template must be a credential-free SSE URL configuration")
    url = urlsplit(entry["url"])
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == 11451
            and url.path == "/mcp/sse" and not url.username and not url.password
            and not url.query and not url.fragment,
            "Published MCP URL must be http://127.0.0.1:11451/mcp/sse without credentials/query/fragment")
    print("PASS published MCP SSE template: loopback default port 11451", flush=True)
    return url


def exercise_server(executable, root, registry, requests, published_url):
    # CI owns an isolated ephemeral fixture. The published host, transport and
    # route were validated above; only its port differs from the default.
    with Server(executable, root, model_yaml(root)) as server:
        try:
            server.wait_ready()
            for route, payload, response_schema in requests:
                status, response = server.request(route, payload)
                require(status == 200, f"{route}: published example returned {status}: {response}")
                validate_instance(response_schema, response, registry, f"{route} actual HTTP 200")
                require(len(response["results"]) == len(payload["images"]),
                        f"{route}: response frame count differs from published input")
                print(f"PASS {route}: exact published payload, HTTP 200, response schema and frame cardinality", flush=True)
            require(published_url.path == "/mcp/sse", "Session helper route differs from published template")
            with Session(server) as session:
                session.initialize()
                response = session.call("tools/list")
                require("error" not in response, f"MCP discovery failed: {response}")
                tools = response["result"]["tools"]
                require({"list_models", "infer_yolo", "infer_ocr"} <= {tool["name"] for tool in tools},
                        "Published MCP connection is missing model/inference tools")
                for tool in tools:
                    Draft202012Validator.check_schema(tool["inputSchema"])
                    require(tool["inputSchema"].get("type") == "object",
                            f"{tool['name']}: tool input schema must describe an object")
                print(f"PASS published MCP SSE connection: initialized, tools/list, {len(tools)} valid input schemas (fixture port {server.port})", flush=True)
        except BaseException:
            print(server.diagnostics(), file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    args = parser.parse_args()
    root, executable = args.project_root.resolve(), args.server.resolve()
    require(executable.is_file(), f"Server executable does not exist: {executable}")
    document = yaml.safe_load((root / "doc/openapi/server.yaml").read_text(encoding="utf-8"))
    registry = validate_openapi(document)
    requests = list(inference_examples(document))
    published_url = mcp_template(root)
    exercise_server(executable, root, registry, requests, published_url)
    print("PASS documentation examples", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
