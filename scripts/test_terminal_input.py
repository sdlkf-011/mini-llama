#!/usr/bin/env python3
"""Exercise UTF-8 input and visible erasure through a real POSIX PTY.

Runs the production chat entry point with the tiny model fixture.
The small screen model covers the cursor/erase sequences emitted by libedit
with TERM=xterm; it checks screen cells, rather than just the submitted bytes.
"""

import argparse
import codecs
import errno
import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import termios
import time
import unicodedata


PROMPT = "mini-llama > "


class Screen:
    def __init__(self, columns):
        self.columns = columns
        self.rows = [[" "] * columns for _ in range(40)]
        self.row = self.column = 0
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    def feed(self, data):
        text = self.pending + self.decoder.decode(data)
        self.pending = ""
        while text:
            char, text = text[0], text[1:]
            if char == "\x1b":
                match = re.match(r"\[([0-9;?]*)([A-Za-z@])", text)
                if not match:
                    self.pending = char + text
                    break
                params, command = match.groups()
                text = text[match.end():]
                values = [int(p or "0") for p in params.lstrip("?").split(";")]
                n = values[0] or 1
                if command == "A":
                    self.row = max(0, self.row - n)
                elif command == "B":
                    self.row += n
                elif command == "C":
                    self.column = min(self.columns - 1, self.column + n)
                elif command == "D":
                    self.column = max(0, self.column - n)
                elif command == "G":
                    self.column = n - 1
                elif command in ("H", "f"):
                    self.row = n - 1
                    self.column = (values[1] or 1) - 1 if len(values) > 1 else 0
                elif command == "K":
                    first = 0 if values[0] in (1, 2) else self.column
                    last = self.column + 1 if values[0] == 1 else self.columns
                    self.rows[self.row][first:last] = [" "] * (last - first)
                elif command not in ("m", "h", "l"):
                    raise AssertionError(f"unsupported terminal sequence: {match[0]!r}")
            elif char == "\r":
                self.column = 0
            elif char == "\n":
                self.row += 1
            elif char == "\b":
                self.column = max(0, self.column - 1)
            elif char == "\a":
                pass
            elif char == "\t":
                self.column = min(self.columns - 1, (self.column // 8 + 1) * 8)
            else:
                width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
                if self.column + width > self.columns:
                    self.row += 1
                    self.column = 0
                self.rows[self.row][self.column] = char
                if width == 2:
                    self.rows[self.row][self.column + 1] = ""
                self.column += width

    def visible(self):
        return "\n".join("".join(row).rstrip() for row in self.rows).rstrip()


class Session:
    def __init__(self, command, columns, locale="C.UTF-8"):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, columns, 0, 0))
        self.settings = termios.tcgetattr(slave)
        self.screen = Screen(columns)
        self.raw = bytearray()
        self.process = subprocess.Popen(
            command, stdin=slave, stdout=slave, stderr=subprocess.DEVNULL,
            env={**os.environ, "TERM": "xterm", "LC_ALL": locale},
        )
        os.close(slave)
        self.wait_for(PROMPT.encode())
        # Observe only the editable region below the startup banner.
        self.screen = Screen(columns)
        self.screen.feed(PROMPT.encode())

    def read(self, timeout):
        if not select.select([self.master], [], [], timeout)[0]:
            return False
        try:
            data = os.read(self.master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                return False
            raise
        self.raw.extend(data)
        self.screen.feed(data)
        return bool(data)

    def wait_for(self, marker):
        deadline = time.monotonic() + 5
        while marker not in self.raw:
            assert time.monotonic() < deadline, bytes(self.raw)
            self.read(0.05)

    def send(self, text):
        os.write(self.master, text.encode())
        # Drain until the editor finishes repainting after this batch of keys.
        while self.read(0.1):
            pass

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
        os.close(self.master)


def check_editor(command, columns, text, erase, expected, locale="C.UTF-8", cursor_text=None):
    session = Session(command, columns, locale)
    try:
        session.send(text)
        if erase:
            session.send("\x7f" * erase)
        visible = session.screen.visible()
        assert visible == (PROMPT + expected).rstrip(), repr(visible)
        cursor_text = expected if cursor_text is None else cursor_text
        total_columns = len(PROMPT) + sum(
            2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in cursor_text
        )
        assert (session.screen.row, session.screen.column) == divmod(total_columns, columns)
        # Submit a real chat command to verify the edited buffer as well as its echo.
        # Empty input must be fully erased; preserved Chinese text is cleared with
        # End + Ctrl-U before submitting /params, avoiding model inference.
        if expected and expected != "/params":
            session.send("\x05\x15")
        if expected != "/params":
            session.send("/params")
        session.send("\n")
        session.wait_for(b"temperature=")
        session.send("\x04")
        assert session.process.wait(timeout=5) == 0
        # Editor teardown must restore canonical input and echo.
        after = termios.tcgetattr(session.master)
        mask = termios.ICANON | termios.ECHO
        assert after[3] & mask == session.settings[3] & mask
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    command = [args.binary, "run", args.model, "--backend", "cpu", "-n", "1"]
    check_editor(command, 80, "你好世界", 4, "")
    check_editor(command, 80, "hello", 5, "")
    check_editor(command, 18, "你好世界" * 6, 24, "")
    check_editor(command, 80, "你好世界", 2, "你好")
    check_editor(command, 80, "你好世界", 4, "", locale="C")
    # Left arrow, deletion and insertion operate on characters rather than bytes.
    check_editor(command, 80, "你好世界\x1b[D\x7f天", 0, "你好天界", cursor_text="你好天")
    check_editor(command, 80, "/par你好ms\x1b[D\x1b[D\x7f\x7fa", 0,
                 "/params", cursor_text="/para")
    result = subprocess.run(
        command, input="\n/params\n/exit".encode(), capture_output=True, check=True,
    )
    output = result.stdout.decode()
    assert output.count(PROMPT) == 3, output
    assert "temperature=" in output and "Bye." in output, output
    assert "\x1b" not in output, repr(output)
    print("terminal UTF-8 input, visible erasure, wrapping, cursor editing and redirected input: passed")


if __name__ == "__main__":
    main()
