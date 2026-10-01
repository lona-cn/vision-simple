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
from test_http_regression import THRESHOLD_INVALID, threshold_expected, threshold_image


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



def expect_post(server, route, expectations, body, *, expected=202, headers=None,
                headers_only=False, abort=False):
    """Use a single 2s deadline for the TCP handshake, final response and EOF."""
    fields = {'Host': f'127.0.0.1:{server.port}', 'Content-Type': 'application/json',
              'Content-Length': str(len(body)), 'Connection': 'keep-alive', **(headers or {})}
    request = f'POST {route} HTTP/1.1\r\n'
    request += ''.join(f'{key}: {value}\r\n' for key, value in fields.items())
    request += ''.join(f'Expect: {value}\r\n' for value in expectations) + '\r\n'
    deadline = time.monotonic() + 2
    buffered = bytearray()
    with socket.create_connection(('127.0.0.1', server.port), timeout=2) as sock:
        def receive():
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'Expect exchange exceeded the 2s deadline')
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(4096)
            except TimeoutError as error:
                raise AssertionError('Expect exchange exceeded the 2s deadline') from error
            require(len(buffered) + len(chunk) <= 65536, 'Unbounded Expect response')
            buffered.extend(chunk)
            return chunk

        def response():
            while b'\r\n\r\n' not in buffered:
                require(receive(), 'Expect connection closed before response headers')
            raw, rest = bytes(buffered).split(b'\r\n\r\n', 1)
            buffered[:] = rest
            lines = raw.decode('iso-8859-1').split('\r\n')
            status = int(lines[0].split()[1])
            response_headers = dict(line.lower().split(':', 1) for line in lines[1:])
            if status == 100:
                return status, response_headers
            require('content-length' in response_headers, 'Expect final response must be length-framed')
            size = int(response_headers['content-length'].strip())
            require(0 <= size <= 65536, 'Unbounded Expect response body')
            while len(buffered) < size:
                require(receive(), 'Expect connection closed before response body')
            del buffered[:size]
            return status, response_headers

        def send(data):
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'Expect exchange exceeded the 2s deadline')
            sock.settimeout(remaining)
            sock.sendall(data)

        send(request.encode('ascii'))
        if not headers_only:
            send(body)
        status, response_headers = response()
        if expected == 100:
            require(status == 100, f'Expect handshake returned {status}, expected 100')
            if abort:
                require(time.monotonic() < deadline, 'Expect exchange exceeded the 2s deadline')
                return
            send(body)
            status, response_headers = response()
            require(status == 202, f'Continued MCP POST returned {status}, expected 202')
        else:
            require(status == expected, f'Expect POST returned {status}, expected {expected}; no interim allowed')
        if status != 202:
            require(response_headers.get('connection', '').strip() == 'close',
                    'Early Expect rejection did not declare connection close')
            try:
                while receive():
                    pass
            except ConnectionResetError:
                # Rejecting an already-uploaded body may close with RST after the full response.
                if headers_only:
                    raise
        require(not buffered, 'Expect exchange returned extra response bytes')
        require(time.monotonic() < deadline, 'Expect exchange exceeded the 2s deadline')


def mcp_expect_matrix(server, session):
    def check(case, route, values, body, **options):
        try:
            expect_post(server, route, values, body, **options)
        except (AssertionError, RuntimeError, OSError) as error:
            raise AssertionError(f'Expect case {case}: {error}') from error

    def recover(case):
        require(session.call('ping').get('result') == {},
                f'Expect case {case}: initialized SSE recovery failed')
        require(not session.pending, f'Expect case {case}: extra SSE RPC result')

    # The length-2 unknown request must be rejected without sending any body.
    rejected = (('unknown-headers-only', ('nonsense',)),
                ('mixed-comma', ('100-continue, nonsense',)),
                ('repeated-comma', ('100-continue, 100-continue',)))
    for case, values in rejected:
        check(case, session.endpoint, values, b'{}', expected=417, headers_only=True)
        recover(case)
    check('unknown-with-body', session.endpoint, ('nonsense',), b'{}', expected=417)
    recover('unknown-with-body')

    # Whitespace-only wire input follows absent/empty flow with this libhv transport.
    accepted = (('absent', (), False), ('empty', ('',), False),
                ('whitespace-only', (' \t ',), False),
                ('continue', ('100-continue',), True),
                ('normalized-continue', (' \t100-CoNtInUe\t ',), True))
    for case, values, handshake in accepted:
        session.counter += 1
        request_id = f'expect-{session.counter}'
        body = json.dumps({'jsonrpc': '2.0', 'method': 'ping', 'id': request_id}).encode()
        check(case, session.endpoint, values, body,
              expected=100 if handshake else 202, headers_only=handshake)
        require(session.receive(request_id).get('result') == {}, f'Expect case {case}: missing SSE RPC result')

    for case, value in (('abort-continue', '100-continue'),
                        ('abort-normalized', ' \t100-CoNtInUe\t ')):
        check(case, session.endpoint, (value,), b'{}', expected=100, headers_only=True, abort=True)
        recover(case)

    # Existing header admission retains precedence over Expect parsing.
    for mode, value in (('unknown', 'nonsense'), ('continue', '100-continue')):
        for guard, headers, expected in (('host', {'Host': 'evil.invalid'}, 403),
                                         ('origin', {'Origin': 'http://evil.invalid'}, 403),
                                         ('media', {'Content-Type': 'text/plain'}, 415),
                                         ('charset', {'Content-Type': 'application/json; charset=latin1'}, 415),
                                         ('length', {'Content-Length': str(64 * 1024 * 1024 + 1)}, 413)):
            case = f'{mode}-{guard}-priority'
            check(case, session.endpoint, (value,), b'{}', expected=expected,
                  headers=headers, headers_only=True)
            recover(case)
        case = f'{mode}-session-priority'
        check(case, '/mcp/messages?session_id=invalid', (value,), b'{}', expected=404, headers_only=True)
        recover(case)
    print('PASS initialized MCP TCP Expect matrix, 2s headers-only rejection/EOF and SSE recovery')


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


def unknown_model_matrix(server, images):
    tasks = ('yolo', 'ocr', 'seg', 'pose', 'obb')
    for kind in tasks:
        for batch in ([], images[:1]):
            status, body = server.request('/v1/infer/' + kind,
                                          {'model': 'missing-model', 'images': batch, 'timeout_ms': 60000})
            require(status == 404 and body['error']['code'] == 'unknown_model'
                    and body['error']['image_index'] is None,
                    f'Native v1 {kind} unknown model: {status} {body}')
        wrong_task_model = 'ppocr-v4' if kind == 'yolo' else 'hd2-fp32'
        status, body = server.request('/v1/infer/' + kind, {'model': wrong_task_model, 'images': []})
        require(status == 404 and body['error']['code'] == 'unknown_model',
                f'Task-mismatched model {kind}: {status} {body}')
        for invalid in ({'model': 'missing-model', 'images': [], 'timeout_ms': 0},
                        {'model': 'missing-model', 'images': [1]},
                        {'model': 'missing-model', 'images': [''] * 129}):
            status, body = server.request('/v1/infer/' + kind, invalid)
            require(status == 400 and body['error']['code'] == 'invalid_request',
                    f'Validation must precede lookup {kind}: {status} {body}')
        status, _, raw = wire(server, 'POST', '/v1/infer/' + kind,
                              {'model': 'missing-model', 'images': []},
                              headers={'Content-Type': 'text/plain'})
        require(status == 415 and json.loads(raw)['error']['code'] == 'unsupported_media_type',
                f'Native content type {kind}: {status} {raw!r}')
    for kind, model in (('yolo', 'hd2-fp32'), ('ocr', 'ppocr-v4')):
        status, body = server.request('/v1/infer/' + kind,
                                      {'model': model, 'images': [], 'timeout_ms': 60000})
        require(status == 200 and body['results'] == [], f'Valid empty batch {kind}: {status} {body}')
    for kind in ('yolo', 'ocr'):
        status, body = server.request('/v0/infer/' + kind, {'model': 'missing-model', 'images': []})
        require(status == 400 and body['error']['code'] == 'unknown_model'
                and body['error']['image_index'] is None, f'Legacy {kind}: {status} {body}')
    for kind in tasks:
        for stream in (False, True):
            status, body = server.request('/v1/chat/completions',
                                          chat_payload(kind, 'missing-model', images[:1], stream=stream))
            require(status == 400 and body['error']['code'] == 'unknown_model'
                    and body['error']['type'] == 'invalid_request_error',
                    f'OpenAI {kind} unknown model: {status} {body}')
    with Session(server) as session:
        session.initialize()
        for kind in tasks:
            response = session.tool('infer_' + kind, {'model': 'missing-model', 'images': images[:1]})
            require('error' not in response and response['result'].get('isError') is True,
                    f'MCP {kind} unknown model must be a tool error: {response}')
            detail = json.loads(response['result']['content'][0]['text'])['error']
            require(detail['code'] == 'unknown_model' and detail['image_index'] is None, str(detail))
    print('PASS unknown-model native v1 404, legacy/OpenAI 400 and MCP tool-error contracts')


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
        mcp_expect_matrix(server, session)
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
        status, busy = server.admin_request('/v0/infer/unload', {'kind': 'ocr', 'model': 'ppocr-v4'})
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


def threshold_protocol_matrix(server):
    shapes = [(64, 64), (128, 128)]
    images = [threshold_image(*shape) for shape in shapes]
    cases = [{}, {'confidence': .1}, {'confidence': .1, 'nms_iou': 0},
             {'confidence': .1, 'nms_iou': 1}, {'confidence': 1}, {'confidence': 0, 'nms_iou': -0.0}]
    for mode in ('raw', 'e2e'):
        for controls in cases:
            payload = chat_payload('yolo', 'tiny-threshold-' + mode, images, **controls)
            expected = threshold_expected(mode, controls, shapes)
            status, body = server.request('/v1/chat/completions', payload)
            require(status == 200 and close_values(json.loads(body['choices'][0]['message']['content']), expected), str(body))
            status, headers, raw = wire(server, 'POST', '/v1/chat/completions', {**payload, 'stream': True})
            require(status == 200 and headers['Content-Type'].startswith('text/event-stream'), str((status, raw)))
            chunks = [line[6:] for line in raw.decode().splitlines() if line.startswith('data: ')]
            require(chunks[-1] == '[DONE]', 'Controlled stream missing terminator')
            content = ''.join(json.loads(chunk)['choices'][0]['delta'].get('content', '') for chunk in chunks[:-1])
            require(close_values(json.loads(content), expected), 'Controlled SSE result differs')
    for version in ('2024-11-05', '2025-11-25'):
        with Session(server) as session:
            session.initialize(version)
            tools = {tool['name']: tool for tool in session.call('tools/list')['result']['tools']}
            for name in ('infer_yolo', 'infer_seg', 'infer_pose', 'infer_obb'):
                require({'confidence', 'nms_iou'} <= tools[name]['inputSchema']['properties'].keys(), name)
            require(not {'confidence', 'nms_iou'} & tools['infer_ocr']['inputSchema']['properties'].keys(), 'OCR schema controls')
            for mode in ('raw', 'e2e'):
                pending = []
                for index, controls in enumerate(cases):
                    request_id = f'{mode}-{index}'
                    session.send('tools/call', {'name': 'infer_yolo', 'arguments': {
                        'model': 'tiny-threshold-' + mode, 'images': images, **controls}}, request_id)
                    pending.append((request_id, threshold_expected(mode, controls, shapes)))
                for request_id, expected in pending:
                    response = session.receive(request_id)
                    require(('structuredContent' in response['result']) == (version != '2024-11-05'), 'MCP negotiated format')
                    require(close_values(tool_data(response), expected), f'MCP controls differ: {request_id}')
            for field in ('confidence', 'nms_iou'):
                for value in THRESHOLD_INVALID:
                    for batch in ([], ['%%%']):
                        response = session.tool('infer_yolo', {'model': 'unknown', 'images': batch, field: value})
                        require('error' not in response and response['result'].get('isError') is True, str(response))
                        detail = json.loads(response['result']['content'][0]['text'])
                        require(detail['error']['code'] == 'invalid_request', str(detail))
                        if 'structuredContent' in response['result']:
                            require(close_values(detail, response['result']['structuredContent']), 'MCP error parity')
                response = session.tool('infer_ocr', {'model': 'unknown', 'images': [], field: .125 if field == 'confidence' else .3})
                require(response['result'].get('isError') is True and
                        json.loads(response['result']['content'][0]['text'])['error']['code'] == 'invalid_request', str(response))
            for field in ('confidence', 'nms_iou'):
                for literal in ('NaN', 'Infinity', '1e400', '-1e400'):
                    session.post(raw='{"jsonrpc":"2.0","id":"syntax","method":"tools/call","params":'
                                     '{"name":"infer_yolo","arguments":{"model":"unknown","images":[],"' +
                                     field + '":' + literal + '}}}')
                    require(session.receive(None)['error']['code'] == -32700, 'Numeric RPC parse contract')
            valid_message = {'jsonrpc': '2.0', 'id': 'overflow', 'method': 'tools/call', 'params': {
                'name': 'infer_yolo', 'arguments': {'model': 'tiny-threshold-raw', 'images': images, 'confidence': .1}}}
            encoded = json.dumps(valid_message)
            malformed = (encoded[:-1] + ',"extension":{"nested":[-1e400]}}',
                         encoded.replace('"confidence": 0.1', '"confidence":1e400,"confidence":0.1'))
            for raw in malformed:
                session.post(raw=raw)
                require(session.receive(None)['error']['code'] == -32700, 'Nested/duplicate overflow escaped RPC parsing')
            string_model = session.tool('infer_yolo', {'model': 'NaN Infinity 1e400', 'images': images[:1]})
            require('error' not in string_model and string_model['result'].get('isError') is True and
                    json.loads(string_model['result']['content'][0]['text'])['error']['code'] == 'unknown_model',
                    f'Numeric-looking model string was treated as JSON syntax: {string_model}')
            require(close_values(tool_data(session.tool('infer_yolo', {
                'model': 'tiny-threshold-raw', 'images': images, 'confidence': .1})),
                threshold_expected('raw', {'confidence': .1}, shapes)), 'MCP parser errors changed valid recovery')
    for field in ('confidence', 'nms_iou'):
        for value in THRESHOLD_INVALID:
            for batch in ([], ['%%%']):
                for stream in (False, True):
                    status, body = server.request('/v1/chat/completions',
                        chat_payload('yolo', 'unknown', batch, stream=stream, **{field: value}))
                    require(status == 400 and body['error']['code'] == 'invalid_request' and
                            body['error']['type'] == 'invalid_request_error', str((status, body)))
        status, body = server.request('/v1/chat/completions',
                                      chat_payload('ocr', 'unknown', images[:1], **{field: .125 if field == 'confidence' else .3}))
        require(status == 400 and body['error']['code'] == 'invalid_request', str((status, body)))
    for field in ('confidence', 'nms_iou'):
        for literal in ('NaN', 'Infinity', '1e400', '-1e400'):
            raw = json.dumps(chat_payload('yolo', 'unknown', images[:1]))[:-1] + ',"' + field + '":' + literal + '}'
            status, body = server.request('/v1/chat/completions', raw=raw)
            require(status == 400 and body['error']['code'] == 'invalid_json', str((status, body)))
    valid_chat = chat_payload('yolo', 'tiny-threshold-raw', images, confidence=.1)
    encoded = json.dumps(valid_chat)
    malformed = (encoded[:-1] + ',"extension":{"nested":[1e400]}}',
                 encoded.replace('"confidence": 0.1', '"confidence":1e400,"confidence":0.1'))
    for raw in malformed:
        status, body = server.request('/v1/chat/completions', raw=raw)
        require(status == 400 and body['error']['code'] == 'invalid_json',
                f'Nested/duplicate overflow escaped OpenAI parsing: {status} {body}')
    status, recovered = server.request('/v1/chat/completions', valid_chat)
    require(status == 200 and close_values(json.loads(recovered['choices'][0]['message']['content']),
            threshold_expected('raw', {'confidence': .1}, shapes)), 'OpenAI parser errors changed valid recovery')
    print('PASS confidence/NMS OpenAI JSON/SSE and both MCP versions, semantic/syntax errors and OCR isolation')


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
            unknown_model_matrix(server, images)
            threshold_protocol_matrix(server)
            baselines = openai_matrix(server, images)
            mcp_matrix(server, images, baselines)
        except BaseException:
            print(server.diagnostics())
            raise
    print('PASS unified protocol regression; all owned connections and processes stopped')


if __name__ == '__main__':
    main()