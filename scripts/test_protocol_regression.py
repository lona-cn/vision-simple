#!/usr/bin/env python3
"""Real HTTP/OpenAI/MCP parity and session-lifecycle regressions (standard library)."""
import argparse
import base64
import contextlib
import http.client
import json
from pathlib import Path
import queue
import socket
import threading
import time

from test_http_regression import Server, close_values, fixture, infer, model_stats, model_yaml, require


def wire(server, method, route, payload=None, headers=None, raw=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=120)
    try:
        connection.request(method, route,
                           body=raw if raw is not None else (json.dumps(payload) if payload is not None else None),
                           headers={'Content-Type': 'application/json', **(headers or {})})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()

def split_post_error(server, route, expected, media='application/json'):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    try:
        connection.putrequest('POST', route)
        connection.putheader('Content-Type', media)
        connection.putheader('Content-Length', '2')
        connection.endheaders()
        time.sleep(.03)  # Let the server process headers before the body exists.
        connection.send(b'{}')
        response = connection.getresponse()
        require(response.status == expected, f'Split upload returned {response.status}, expected {expected}')
        response.read()
    finally:
        connection.close()


def oversized_headers(server, route):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    try:
        connection.putrequest('POST', route)
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(64 * 1024 * 1024 + 1))
        connection.endheaders()  # Rejection must not wait for or allocate the body.
        response = connection.getresponse()
        require(response.status == 413, f'Oversized headers returned {response.status}')
        response.read()
    finally:
        connection.close()



class Session:
    def __init__(self, server):
        self.server = server
        self.connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=120)
        self.events = queue.Queue()
        self.pending = {}
        self.response = self.sock = self.thread = None
        self.counter = 0

    def __enter__(self):
        try:
            self.connection.request('GET', '/mcp/sse', headers={'Accept': 'text/event-stream'})
            self.sock = self.connection.sock
            self.response = self.connection.getresponse()
            require(self.response.status == 200, f'SSE handshake failed: {self.response.status}')
            require(self.response.getheader('Content-Type', '').startswith('text/event-stream'), 'SSE content type')
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            kind, data = self.event()
            require(kind == 'endpoint' and data.startswith('/mcp/messages?'), f'Endpoint event: {(kind, data)}')
            self.endpoint = data
            return self
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            kind, data = 'message', []
            while True:
                line = self.response.readline()
                if not line:
                    self.events.put(('closed', ''))
                    return
                text = line.decode('utf-8').rstrip('\r\n')
                if not text:
                    if data:
                        self.events.put((kind, '\n'.join(data)))
                    kind, data = 'message', []
                elif text.startswith('event:'):
                    kind = text[6:].lstrip(' ')
                elif text.startswith('data:'):
                    data.append(text[5:].lstrip(' '))
                elif text.startswith(':'):
                    self.events.put(('heartbeat', text))
        except (OSError, ValueError) as error:
            self.events.put(('closed', str(error)))

    def event(self, timeout=30):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty as error:
            raise AssertionError('Timed out waiting for SSE event') from error

    def post(self, message=None, raw=None):
        status, _, body = wire(self.server, 'POST', self.endpoint, message, raw=raw)
        require(status == 202, f'MCP POST expected 202, got {status}: {body[:500]!r}')

    def send(self, method, params=None, request_id=None):
        message = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            message['params'] = params
        if request_id is not None:
            message['id'] = request_id
        self.post(message)

    def receive(self, request_id, timeout=120):
        key = json.dumps(request_id)
        if key in self.pending:
            return self.pending.pop(key)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            kind, data = self.event(max(.01, deadline - time.monotonic()))
            if kind == 'heartbeat':
                continue
            require(kind == 'message', f'Expected message, got {(kind, data)}')
            response = json.loads(data)
            require(response.get('jsonrpc') == '2.0' and 'id' in response, f'Invalid RPC response: {response}')
            other = json.dumps(response['id'])
            if other == key:
                return response
            self.pending[other] = response
        raise AssertionError(f'No response for {request_id!r}')

    def call(self, method, params=None):
        self.counter += 1
        request_id = f'call-{self.counter}'
        self.send(method, params, request_id)
        return self.receive(request_id)

    def initialize(self, version='2025-11-25'):
        result = self.call('initialize', {'protocolVersion': version, 'capabilities': {},
                                         'clientInfo': {'name': 'vision-regression', 'version': '1'}})
        require(result['result']['protocolVersion'] == version, f'Negotiation failed: {result}')
        require('tools' in result['result']['capabilities'], 'Missing tools capability')
        self.send('notifications/initialized')
        return result

    def tool(self, name, arguments):
        return self.call('tools/call', {'name': name, 'arguments': arguments})

    def close(self):
        if self.sock:
            with contextlib.suppress(OSError):
                self.sock.shutdown(socket.SHUT_RDWR)
        self.connection.close()
        if self.thread:
            self.thread.join(timeout=5)
            require(not self.thread.is_alive(), 'SSE reader failed to stop')
        if self.response:
            self.response.close()

    def __exit__(self, *_):
        self.close()


def tool_data(response):
    require('result' in response and not response['result'].get('isError'), f'Tool failed: {response}')
    result = response['result']
    text = json.loads(result['content'][0]['text'])
    if 'structuredContent' in result:
        require(close_values(text, result['structuredContent']), 'MCP text/structured results differ')
    return text


def wait_activity(server, active):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        models = model_stats(server).values()
        count = sum(item['active_requests'] for item in models)
        if (count > 0) == active:
            return
        time.sleep(.02)
    raise AssertionError(f'Model activity failed to become {active}: {models}')


def chat_payload(kind, model, images, **controls):
    return {'model': f'{kind}:{model}', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image}'}}
        for image in images]}], **controls}


def openai_matrix(server, images):
    split_post_error(server, '/v1/chat/completions', 415, 'text/plain')
    oversized_headers(server, '/v1/chat/completions')
    status, first = server.request('/v1/models?limit=1', method='GET')
    require(status == 200 and first['object'] == 'list', f'Model list: {first}')
    collected = list(first['data'])
    while first.get('has_more'):
        status, first = server.request('/v1/models?limit=1&after=' + first['next_cursor'], method='GET')
        require(status == 200, f'Model pagination: {first}')
        collected += first['data']
    ids = [item['id'] for item in collected]
    require(ids == sorted(set(ids)) and {'yolo:hd2-fp32', 'yolo:hd2-fp16', 'ocr:ppocr-v4'} <= set(ids),
            f'Incomplete or duplicated catalog: {ids}')
    for query in ('limit=0', 'limit=201', 'after=not-a-cursor'):
        status, body = server.request('/v1/models?' + query, method='GET')
        require(status == 400 and body['error']['type'] == 'invalid_request_error', str(body))
    baselines = {}
    for kind, model in (('yolo', 'hd2-fp32'), ('ocr', 'ppocr-v4')):
        baseline = infer(server, kind, model, images)
        baselines[kind] = baseline
        payload = chat_payload(kind, model, images)
        status, response = server.request('/v1/chat/completions', payload)
        require(status == 200 and response['object'] == 'chat.completion', str(response))
        require(close_values(json.loads(response['choices'][0]['message']['content']), baseline), 'OpenAI/v0 results differ')
        status, headers, raw = wire(server, 'POST', '/v1/chat/completions', {**payload, 'stream': True})
        require(status == 200 and headers['Content-Type'].startswith('text/event-stream'), str((status, headers)))
        chunks = [line[6:] for line in raw.decode().splitlines() if line.startswith('data: ')]
        require(chunks[-1] == '[DONE]', f'Missing stream terminator: {chunks[-1:]}')
        chunks = [json.loads(chunk) for chunk in chunks[:-1]]
        require(len({chunk['id'] for chunk in chunks}) == 1, 'Streaming ID changed')
        require(chunks[0]['choices'][0]['delta']['role'] == 'assistant', 'Missing role delta')
        require(chunks[-1]['choices'][0]['finish_reason'] == 'stop', 'Missing finish reason')
        content = ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks)
        require(close_values(json.loads(content), baseline), 'OpenAI streaming/v0 results differ')
        status, response = server.request('/v1/chat/completions', chat_payload(kind, model, ['%%%'], stream=True))
        require(status == 400 and response['error']['code'] == 'invalid_image', str(response))
    invalid = [chat_payload('yolo', 'hd2-fp32', images, stream='true'),
               chat_payload('yolo', 'hd2-fp32', images, temperature=.5),
               chat_payload('yolo', 'hd2-fp32', [], n=2),
               {'model': 'yolo:hd2-fp32', 'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'http://127.0.0.1/private'}}]}]}]
    for payload in invalid:
        status, response = server.request('/v1/chat/completions', payload)
        require(status == 400 and response['error']['type'] == 'invalid_request_error', str(response))
    print('PASS OpenAI discovery, cursor pagination, JSON/SSE structured parity and error boundaries')
    return baselines


def mcp_matrix(server, images, baselines):
    for method, path in (('GET', '/mcp/sse'), ('POST', '/mcp/messages?session_id=invalid')):
        for headers in ({'Origin': 'http://evil.invalid'}, {'Host': 'evil.invalid'}):
            status, _, _ = wire(server, method, path, {}, headers=headers)
            require(status in (400, 403, 421), f'Unsafe MCP origin/host accepted: {status}')
    with Session(server) as session:
        oversized_headers(server, session.endpoint)
        premature = session.call('tools/list')
        require('error' in premature, f'Uninitialized tools request accepted: {premature}')
        session.initialize()
        tools = session.call('tools/list')['result']['tools']
        require({'list_models', 'infer_yolo', 'infer_ocr'} <= {tool['name'] for tool in tools},
                f'Legacy tool capabilities disappeared: {tools}')
        for tool in tools:
            require(tool['inputSchema']['type'] == 'object', 'Input schema must describe an object')
        page = tool_data(session.tool('list_models', {'limit': 1}))
        models = list(page['data'])
        while page.get('next_cursor'):
            page = tool_data(session.tool('list_models', {'limit': 1, 'cursor': page['next_cursor']}))
            models += page['data']
        require(len({model['id'] for model in models}) == len(models) and len(models) >= 3, str(models))
        for kind, model in (('yolo', 'hd2-fp32'), ('ocr', 'ppocr-v4')):
            response = session.tool('infer_' + kind, {'model': model, 'images': images})
            require('structuredContent' in response['result'], 'New protocol must return structured content')
            require(close_values(tool_data(response), baselines[kind]), 'MCP/v0 results differ')
        error = session.tool('infer_yolo', {'model': 'hd2-fp32', 'images': [images[0], '%%%']})
        require(error['result']['isError'], str(error))
        detail = json.loads(error['result']['content'][0]['text'])['error']
        require(detail['code'] == 'invalid_image' and detail['image_index'] == 1, str(detail))
        session.send('ping', request_id=1)
        session.send('ping', request_id='1')
        require(type(session.receive(1)['id']) is int and type(session.receive('1')['id']) is str,
                'Numeric and string request IDs collided')
        require(session.call('unknown')['error']['code'] == -32601, 'Unknown RPC method mapping')
        session.send('notifications/unrecognized')
        require('result' in session.call('ping'), 'Unknown notification broke session')
        session.post(raw='{')
        require(session.receive(None)['error']['code'] == -32700, 'Malformed JSON mapping')
        for value in ([], {'jsonrpc': '2.0', 'method': 'ping', 'id': True}):
            session.post(value)
            require(session.receive(None)['error']['code'] == -32600, 'Malformed request mapping')
        session.send('tools/call', {'name': 'infer_ocr', 'arguments': {
            'model': 'ppocr-v4', 'images': [images[1]] * 16}}, 'cancel-me')
        wait_activity(server, True)
        status, busy = server.request('/v0/infer/unload', {'kind': 'ocr', 'model': 'ppocr-v4'})
        require(status == 409 and busy['error']['code'] == 'model_busy', str(busy))
        session.send('notifications/cancelled', {'requestId': 'cancel-me', 'reason': 'regression'})
        cancelled = session.receive('cancel-me')
        if 'result' in cancelled:
            require(cancelled['result'].get('isError'), f'Cancelled request succeeded: {cancelled}')
            detail = json.loads(cancelled['result']['content'][0]['text'])['error']
            require(detail['code'] == 'request_cancelled', str(detail))
        else:
            require(cancelled['error']['code'] == -32800, str(cancelled))
        wait_activity(server, False)
        tool_data(session.tool('infer_ocr', {'model': 'ppocr-v4', 'images': [images[1]]}))
        timeout = session.tool('infer_ocr', {'model': 'ppocr-v4', 'images': [images[1]], 'timeout_ms': 1})
        require(timeout['result']['isError'], str(timeout))
        require(json.loads(timeout['result']['content'][0]['text'])['error']['code'] == 'request_timeout', str(timeout))
        # A different session cannot cancel an equal request ID in this session.
        session.send('tools/call', {'name': 'infer_ocr', 'arguments': {
            'model': 'ppocr-v4', 'images': [images[1]] * 4}}, 'isolated')
        wait_activity(server, True)
        with Session(server) as other:
            other.initialize()
            other.send('notifications/cancelled', {'requestId': 'isolated'})
            require('result' in other.call('ping'), 'Second session became unusable')
        result = tool_data(session.receive('isolated'))
        expected = infer(server, 'ocr', 'ppocr-v4', [images[1]])
        require(close_values(result, {'results': expected['results'] * 4}), 'Cross-session cancellation changed results')
        endpoint = session.endpoint
        session.send('tools/call', {'name': 'infer_ocr', 'arguments': {
            'model': 'ppocr-v4', 'images': [images[1]] * 16}}, 'disconnect-me')
        wait_activity(server, True)
    wait_activity(server, False)
    status, _, _ = wire(server, 'POST', endpoint, {'jsonrpc': '2.0', 'id': 1, 'method': 'ping'})
    require(status == 404, f'Disconnected session remained usable: {status}')
    split_post_error(server, endpoint, 404)
    with Session(server) as legacy:
        legacy.initialize('2024-11-05')
        require(close_values(tool_data(legacy.tool('infer_yolo', {'model': 'hd2-fp32', 'images': images})), baselines['yolo']), 'Legacy MCP/v0 results differ')
    print('PASS MCP lifecycle, discovery, real tool parity, errors, typed IDs, cancellation and disconnect isolation')
    with contextlib.ExitStack() as stack:
        for _ in range(32):
            stack.enter_context(Session(server))
        status, _, _ = wire(server, 'GET', '/mcp/sse')
        require(status == 503, f'MCP session bound was not enforced: {status}')
    with Session(server) as recovered:
        recovered.initialize()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            kind, _ = recovered.event(max(.01, deadline - time.monotonic()))
            if kind == 'heartbeat':
                break
        else:
            raise AssertionError('No MCP keep-alive comment')
        require('result' in recovered.call('ping'), 'Session failed after heartbeat')
    print('PASS early body limits, split-upload error responses, session bound/recovery and keep-alive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    images = [base64.b64encode(fixture(root, path).read_bytes()).decode('ascii')
              for path in ('app/assets/test/hd2.png', 'doc/images/ppocr.png')]
    with Server(args.server.resolve(), root, model_yaml(root)) as server:
        try:
            server.wait_ready()
            baselines = openai_matrix(server, images)
            mcp_matrix(server, images, baselines)
        except BaseException:
            print(server.diagnostics())
            raise
    print('PASS unified protocol regression; all owned connections and processes stopped')


if __name__ == '__main__':
    main()
