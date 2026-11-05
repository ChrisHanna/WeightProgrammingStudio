"""Fixed local Engine adapter; no external network, pulls or guest commands.

Docker's Go CLI reserves >256 MiB virtual memory before doing any work. This
stdlib adapter fits the dedicated controller's inherited hard RLIMIT_AS. Rust
retains its bounded readers, deadline/revocation polling and native cleanup.
Only the launcher's finite command vocabulary is accepted, never shell input.
"""
import http.client
import json
import os
import socket
import struct
import sys
from urllib.parse import quote, urlencode

MAX = 262144
SOCKET = "/var/run/docker.sock"


class Engine(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=10)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(SOCKET)


def request(method, path, body=None):
    connection = Engine()
    try:
        data = json.dumps(body).encode() if body is not None else b""
        connection.request(method, path, data, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(MAX + 1)
        if len(raw) > MAX:
            raise ValueError("Engine control output exceeded")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Engine status {response.status}: {raw[:1024]!r}")
        return json.loads(raw) if raw else None
    finally:
        connection.close()


def resource(name):
    return "/containers/" + quote(name, safe="")


def create(args):
    values = {}
    index = 0
    while index < len(args) and args[index].startswith("--"):
        flag = args[index]
        if "=" in flag:
            key, value = flag.split("=", 1)
            index += 1
        elif flag in ("--init", "--read-only"):
            key, value = flag, True
            index += 1
        else:
            key, value = flag, args[index + 1]
            index += 2
        values.setdefault(key, []).append(value)
    expected = {"--pull", "--init", "--name", "--label", "--network", "--read-only",
                "--cap-drop", "--cap-add", "--security-opt", "--pids-limit", "--memory",
                "--memory-swap", "--cpus", "--ulimit", "--stop-timeout", "--log-driver",
                "--shm-size", "--tmpfs", "--mount", "--env", "--workdir", "--entrypoint"}
    if set(values) != expected or values["--pull"] != ["never"]:
        raise ValueError("unsupported fixed Engine create profile")
    one = lambda key: values[key][0]
    mounts = []
    for value in values["--mount"]:
        fields = dict(part.split("=", 1) if "=" in part else (part, True)
                      for part in value.split(","))
        if set(fields) != {"type", "src", "dst", "readonly"} or fields["type"] != "bind":
            raise ValueError("unsupported Engine mount")
        mounts.append({"Type": "bind", "Source": fields["src"],
                       "Target": fields["dst"], "ReadOnly": True})
    ulimits = []
    for value in values["--ulimit"]:
        name, bounds = value.split("=", 1)
        soft, hard = bounds.split(":")
        ulimits.append({"Name": name, "Soft": int(soft), "Hard": int(hard)})
    tmpfs = dict(value.split(":", 1) for value in values["--tmpfs"])
    body = {"Image": args[index], "Cmd": args[index + 1:],
            "Entrypoint": [one("--entrypoint")], "WorkingDir": one("--workdir"),
            "Env": values["--env"], "Labels": dict(value.split("=", 1) for value in values["--label"]),
            "AttachStdout": True, "AttachStderr": True, "Tty": False,
            "StopTimeout": int(one("--stop-timeout")),
            "HostConfig": {"Init": True, "NetworkMode": one("--network"),
                           "ReadonlyRootfs": True, "Privileged": False,
                           "CapDrop": values["--cap-drop"],
                           "CapAdd": sorted("CAP_" + cap.removeprefix("CAP_") for cap in values["--cap-add"]),
                           "SecurityOpt": values["--security-opt"],
                           "PidsLimit": int(one("--pids-limit")), "Memory": int(one("--memory")),
                           "MemorySwap": int(one("--memory-swap")),
                           "NanoCpus": int(float(one("--cpus")) * 1000000000),
                           "Ulimits": ulimits, "LogConfig": {"Type": one("--log-driver")},
                           "ShmSize": int(one("--shm-size")), "Tmpfs": tmpfs, "Mounts": mounts}}
    result = request("POST", "/containers/create?" + urlencode({"name": one("--name")}), body)
    # This trusted helper can outlive a crashed controller. Persist server
    # completion BEFORE replying so recovery never races a deferred create.
    guards = [mount["Source"] for mount in mounts if mount["Target"] == "/guard"]
    if len(guards) != 1:
        raise ValueError("exact private create journal required")
    journal = os.path.dirname(guards[0])
    if os.path.basename(journal) != one("--name"):
        raise ValueError("create journal differs from owned name")
    fd = os.open(journal + "/create_terminal.json",
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as marker:
        marker.write(json.dumps({"schema_version": 1, "id": result["Id"]}).encode())
        marker.flush()
        os.fsync(marker.fileno())
    directory = os.open(journal, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(result["Id"])


def start_attached(name):
    # Attach BEFORE start, including fast exits. Engine's non-TTY stream has
    # an authenticated local transport header, not guest-controlled framing.
    stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stream.settimeout(10)
    try:
        stream.connect(SOCKET)
        path = resource(name) + "/attach?stream=1&stdout=1&stderr=1"
        stream.sendall((f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                        "Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: 0\r\n\r\n").encode())
        header = bytearray()
        while not header.endswith(b"\r\n\r\n"):
            byte = stream.recv(1)
            if not byte or len(header) >= 32768:
                raise ValueError("invalid bounded Engine attach header")
            header.extend(byte)
        if header.split(b"\r\n", 1)[0].split()[1] != b"101":
            raise ValueError("Engine attach upgrade required")
        request("POST", resource(name) + "/start")
        outputs = {1: sys.stdout.buffer, 2: sys.stderr.buffer}
        count = {1: 0, 2: 0}
        def exact(size):
            data = bytearray()
            while len(data) < size:
                part = stream.recv(size - len(data))
                if not part:
                    if not data:
                        return None
                    raise ValueError("truncated Engine stream")
                data.extend(part)
            return bytes(data)
        while True:
            frame = exact(8)
            if frame is None:
                break
            channel, padding, size = struct.unpack(">B3sI", frame)
            if channel not in outputs or padding != b"\0\0\0" or size > MAX - count[channel]:
                raise ValueError("invalid or exceeded Engine stream")
            payload = exact(size) if size else b""
            if payload is None:
                raise ValueError("missing Engine payload")
            count[channel] += size
            outputs[channel].write(payload)
            outputs[channel].flush()
    finally:
        stream.close()


def main(args):
    operation = args[0]
    if operation == "version" and args[1:] == ["--format", "{{json .Server}} "]:
        print(json.dumps(request("GET", "/version")))
    elif operation == "info" and len(args) == 3 and args[1] == "--format":
        info = request("GET", "/info")
        print(json.dumps({key: info[key] for key in
                          ("CgroupVersion", "MemoryLimit", "PidsLimit", "SecurityOptions")}))
    elif operation == "image" and len(args) == 3 and args[1] == "inspect":
        print(json.dumps([request("GET", "/images/" + quote(args[2], safe="") + "/json")]))
    elif operation == "create":
        create(args[1:])
    elif operation == "start" and len(args) == 3 and args[1] == "--attach":
        start_attached(args[2])
    elif operation == "inspect" and len(args) == 2:
        print(json.dumps([request("GET", resource(args[1]) + "/json")]))
    elif operation == "rm" and len(args) == 3 and args[1] == "--force":
        request("DELETE", resource(args[2]) + "?force=1")
        print(args[2])
    elif operation == "ps" and len(args) == 6 and args[1:3] == ["--all", "--filter"]:
        key, value = args[3].split("=", 1)
        if key != "name" or args[4:] != ["--format", "{{.ID}} "]:
            raise ValueError("unsupported Engine inventory query")
        rows = request("GET", "/containers/json?" + urlencode({"all": "1", "filters": json.dumps({key: [value]})}))
        for row in rows:
            print(row["Id"])
    else:
        raise ValueError("unsupported finite Engine operation")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, http.client.HTTPException) as error:
        print(str(error)[:2048], file=sys.stderr)
        sys.exit(1)
