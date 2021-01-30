from collections import deque

class Display:
    def __init__(self, *lines: str, num_lines: int = None, width: int = None):
        self._num_lines = num_lines
        self.lines = deque(maxlen=self._num_lines)
        self.feedlines(*lines)

    def __str__(self):
        try:
            return '\n'.join(self.lines) 
        except TypeError:
            raise Exception(self.lines)

    def feedlines(self, *lines: str):
        for line in lines:
            self.lines.append(line)

    def setlines(self, *lines):
        self.lines = deque(lines, maxlen=self._num_lines)

    def clear(self):
        self.lines = deque(maxlen=self._num_lines)

class ConsoleInterface:
    """ Formats its own data and nicely displays them to console """
    def __str__(self):
        return '\n'.join([str(disp) for disp in self.displays])

    def __init__(self, *displays):
        self.displays = list(displays)
    def add(self, display):
        self.displays.append(display)


if __name__ == "__main__":
    d = Display("line", "line2")
    print(d.lines)
