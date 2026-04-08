from functools import lru_cache
from multiprocessing import Pool
from threading import Thread
import time
from typing import Callable
from PIL import Image as PILImage
from blessed import Terminal
import numpy as np
from tools import get_frames
import sys
import io

class Color:

    __slots__ = ('_v',)

    def __init__(self, r=0, g=0, b=0):
        self._v = np.array([r, g, b], dtype=np.int16)

    @classmethod
    def from_array(cls, arr):
        if isinstance(arr, Color):
            return cls(*arr._v)
        return cls(*arr)

    def __add__(self, other):
        o = self._to_vector(other)
        return Color.from_array(self._v + o)

    def __sub__(self, other):
        o = self._to_vector(other)
        return Color.from_array(self._v - o)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return Color.from_array(self._v * value)
        o = self._to_vector(value)
        return Color.from_array(self._v * o)

    def __truediv__(self, value):
        if isinstance(value, (int, float)):
            return Color.from_array(self._v / value)
        o = self._to_vector(value)
        return Color.from_array(self._v / o)

    def __iter__(self):
        return iter(self._v.tolist())

    def __repr__(self):
        r, g, b = self._v
        return f"Color(r={r}, g={g}, b={b})"

    def clamp(self, minv=(0, 0, 0), maxv=(255, 255, 255)):

        minv = np.array(minv, dtype=np.int16)
        maxv = np.array(maxv, dtype=np.int16)
        return Color.from_array(np.clip(self._v, minv, maxv))

    def to_tuple(self):
        return tuple(self._v.tolist())

    def copy(self):
        return Color.from_array(self._v.copy())

    @staticmethod
    def _to_vector(value):
        if isinstance(value, Color):
            return value._v
        return np.array(value, dtype=np.int16)

class Pixel:

    def __init__(
            self,
            char: str = " ",
            color: Color = Color(255, 255, 255),
            color_bg: Color = Color(0, 0, 0)
    ):
        self.char  = char
        self.color = color
        self.color_bg = color_bg

class FrameBuffer:

    def __init__(
            self,
            init_width: int,
            init_height: int,
            color_mask_max: Color = Color(255, 255, 255),
            color_mask_min: Color = Color(0, 0, 0),
            color_default: Color = Color(0, 0, 0),
            color_default_bg: Color = Color(0, 0, 0)
    ):
        self.buffer = [Pixel() for _ in range(init_width * init_height)]
        self.width = init_width
        self.height = init_height
        self.bufflen = self.width * self.height

        self.color_mask_max = color_mask_max
        self.color_mask_min = color_mask_min
        
        self.color_default = color_default
        self.color_default_bg = color_default_bg
    
    def write(
            self,
            content: list,
            fromidx: int
    ):
        self.buffer[fromidx:fromidx+len(content)] = content

    def get(self):
        return self.buffer.copy()
    
    def changewh(
            self,
            width: int | None = None,
            height: int | None = None
    ):
        self.width = width if width is not None else self.width
        self.height = height if height is not None else self.height
        self.bufflen = len(self.buffer)
        actuallen = self.width * self.height
        if self.bufflen != actuallen:
            diff: int = actuallen - self.bufflen
            if diff > 0: self.buffer.extend([Pixel()] * diff)
            else: self.buffer = self.buffer[:actuallen]
    
    def clean(self):  self.buffer = [Pixel() for _ in range(self.width * self.height)]

class RelativeSize:

    def __init__(
        self,
        percent_w: int,
        percent_h: int
    ):
        self.coeff_w = percent_w / 100
        self.coeff_h = percent_h / 100

    @lru_cache
    def get_size(
        self,
        width: int,
        height: int
    ):
        return (int(self.coeff_w * width), int(self.coeff_h * height))

class RelativePosition:

    def __init__(
        self,
        percent_x: int,
        percent_y: int
    ):
        self.coeff_x = percent_x / 100
        self.coeff_y = percent_y / 100
    
    @lru_cache
    def get_pos(
        self,
        width: int,
        height: int
    ):
        return (int(self.coeff_x * width), int(self.coeff_y * height))

class OverlayTexture:

    def __init__(
            self,
            size: RelativeSize,
            position: RelativePosition,
            color_offset: Color = Color(0, 0, 0),
            color_bg_offset: Color = Color(0, 0, 0)
    ):
        self.size = size
        self.pos = position
        self.color_offset = color_offset
        self.color_bg_offset = color_bg_offset
    
    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        w, h = self.size.get_size(width, height)

        for y1 in range(y, y+h):
            framebuff.write(
                [
                    Pixel(
                        " ",
                        buff[width*y1+x].color+self.color_offset,
                        buff[width*y1+x].color_bg+self.color_bg_offset
                    )
                ] * w, x+width*(y1-1)
            )

class StaticTexture:
    def __init__(
            self,
            size: RelativeSize,
            position: RelativePosition,
            color: Color | None = None,
            color_bg: Color | None = None
    ):
        self.size = size
        self.pos = position
        self.color = color
        self.color_bg = color_bg
    
    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        w, h = self.size.get_size(width, height)

        for y1 in range(y, y+h):
            framebuff.write(
                [
                    Pixel(
                        " ",
                        buff[width*y1+x].color if self.color is None else self.color,
                        buff[width*y1+x].color_bg if self.color_bg is None else self.color_bg
                    )
                ] * w, x+width*(y1-1)
            )

class DataTexture:

    def __init__(
            self,
            size: RelativeSize,
            position: RelativePosition,
            data: list[Pixel]
    ):
        self.size = size
        self.pos = position
        self.data = data
    
    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        datalen = len(self.data)

        x, y = self.pos.get_pos(width, height)
        w, h = self.size.get_size(width, height)
        if datalen < width*height: return
        
        for y1 in range(y, y+h):
            framebuff.write(
                [
                    self.data[w*(y1-y)+(_-x)]
                for _ in range(x, x+w)], x+width*(y1-1)
            )

class OverlayLabel:

    def __init__(
            self,
            content: str,
            position: RelativePosition,
            color_offset: Color = Color(255, 255, 255),
            color_bg_offset: Color = Color(0, 0, 0),
            centered: bool = False
    ):
        self.content = content
        self.pos = position
        self.color_offset = color_offset
        self.color_bg_offset = color_bg_offset
        self.centered = centered
    
    def changec(self, content: str): self.content = content

    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        x = x - len(self.content)//2 if self.centered else x

        framebuff.write([Pixel(ch, buff[width*y+x+c].color+self.color_offset, buff[width*y+x+c].color_bg+self.color_bg_offset) for c, ch in enumerate(self.content)], width*y+x)

class StaticLabel:

    def __init__(
            self,
            content: str,
            position: RelativePosition,
            color: Color | None = None,
            color_bg: Color | None = None,
            centered: bool = False
    ):
        self.content = content
        self.pos = position
        self.color = color
        self.color_bg = color_bg
        self.centered = centered
    
    def changec(self, content: str): self.content = content

    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        x = x - len(self.content)//2 if self.centered else x

        framebuff.write([Pixel(ch, buff[width*y+x+c].color if self.color is None else self.color, buff[width*y+x+c].color_bg if self.color_bg is None else self.color_bg) for c, ch in enumerate(self.content)], width*y+x)

class OverlayInputLabel:

    def __init__(
            self,
            prompt: str,
            position: RelativePosition,
            color_offset: Color = Color(255, 255, 255),
            color_bg_offset: Color = Color(0, 0, 0),
            formatting: str = "{prompt}: {content}",
            init_input_content: str = "",
            input_rules: list[Callable] = [],
            centered: bool = False
    ):
        self.prompt = prompt
        self.input_content = init_input_content
        self.pos = position
        self.color_offset = color_offset
        self.color_bg_offset = color_bg_offset
        self.formatting = formatting
        self.centered = centered
        self.rules = input_rules
        self.content = self.formatting.format_map({"prompt" : self.prompt, "content" : self.input_content})
    
    def changep(self, prompt: str):
        self.prompt = prompt
        self.content = self.formatting.format_map({"prompt" : self.prompt, "content" : self.input_content})

    def append(self, input: str):
        inp = self.input_content + input
        if False in [rule(inp) for rule in self.rules]: return
        self.input_content = inp
        self.content = self.formatting.format_map({"prompt" : self.prompt, "content" : self.input_content})
    
    def pop(self, index: int = -1):
        self.input_content = self.input_content[:index]
        self.content = self.formatting.format_map({"prompt" : self.prompt, "content" : self.input_content})

    def popsplit(self, index: int = -1, sep: str = " "):
        self.input_content = sep.join(self.input_content.split(sep)[:index])
        self.content = self.formatting.format_map({"prompt" : self.prompt, "content" : self.input_content})

    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        x = x - len(self.content)//2 if self.centered else x

        framebuff.write([Pixel(ch, buff[width*y+x+c].color+self.color_offset, buff[width*y+x+c].color_bg+self.color_bg_offset) for c, ch in enumerate(self.content)], width*y+x)

class StaticInputLabel:

    def __init__(
            self,
            prompt: str,
            position: RelativePosition,
            color: Color | None = None,
            color_bg: Color | None = None,
            formatting: str = "{prompt}: {content}",
            init_input_content: str = "",
            input_rules: list[Callable] = [],
            centered: bool = False
    ):
        self.prompt = prompt
        self.input_content = init_input_content
        self.pos = position
        self.color = color
        self.color_bg = color_bg
        self.formatting = formatting
        self.centered = centered
        self.rules = input_rules
        self.content = self.formatting.format(prompt=self.prompt, content=self.input_content)
    
    def changep(self, prompt: str):
        self.prompt = prompt
        self.content = self.formatting.format(prompt=self.prompt, content=self.input_content)

    def append(self, input: str):
        inp = self.input_content + input
        if False in [rule(inp) for rule in self.rules]: return
        self.input_content = inp
        self.content = self.formatting.format(prompt=self.prompt, content=self.input_content)
    
    def pop(self, index: int = -1):
        self.input_content = self.input_content[:index]
        self.content = self.formatting.format(prompt=self.prompt, content=self.input_content)

    def popsplit(self, index: int = -1, sep: str = " "):
        self.input_content = sep.join(self.input_content.split(sep)[:index])
        self.content = self.formatting.format(prompt=self.prompt, content=self.input_content)

    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height
        buff = framebuff.get()

        x, y = self.pos.get_pos(width, height)
        x = x - len(self.content)//2 if self.centered else x

        framebuff.write([Pixel(ch, buff[width*y+x+c].color if self.color is None else self.color, buff[width*y+x+c].color_bg if self.color_bg is None else self.color_bg) for c, ch in enumerate(self.content)], width*y+x)

class Image:

    def __init__(
            self,
            data: bytes | PILImage.Image,
            frompil: bool = False,
            max_size: tuple[int, int] = (100, 100)
    ):
        self.data = data if frompil else PILImage.open(io.BytesIO(data)).convert("RGB")
        self.data = self.data.resize(max_size)

    @lru_cache
    def resize(self, width: int, height: int):
        return list(self.data.resize((width, height)).getdata())
    
    @staticmethod
    def from_file(file: str):
        with open(file, "rb") as f: data = f.read()
        return Image(data)

class ImageTexture:

    def __init__(
            self,
            size: RelativeSize,
            position: RelativePosition,
            image: Image
    ):
        self.size = size
        self.pos = position
        self.image = image

    @staticmethod
    def load_animation(file: str):
        pilframes = get_frames(file)
        return [
            ImageTexture(
                RelativeSize(0, 0),
                RelativePosition(0, 0),
                Image(x, True)
            ) 
            for x in pilframes
        ]

    def _process(self, framebuff: FrameBuffer):
        width = framebuff.width
        height = framebuff.height

        x, y = self.pos.get_pos(width, height)
        w, h = self.size.get_size(width, height)
        data = self.image.resize(w, h)

        for y1 in range(y, y+h):
            framebuff.write(
                [
                    Pixel(
                        " ",
                        Color(*data[w*(y1-y)+(_-x)]),
                        Color(*data[w*(y1-y)+(_-x)])
                    )
                for _ in range(x, x+w)], x+width*(y1-1)
            )

class Animation:

    def __init__(
            self,
            size: RelativeSize,
            position: RelativePosition,
            frames: list[ImageTexture],
            fps: int = 10,
            speed: float = 1.0
    ):
        self.size = size
        self.pos = position
        self.frames = frames
        for frame in self.frames:
            frame.pos = self.pos
            frame.size = self.size
        self.fps = fps
        self.speed = speed
        self.state = 0
        self.lastframe = time.time()
    
    @property
    def frame_time(self): return 1/(self.fps*self.speed)

    def _process(self, framebuff: FrameBuffer):
        nowt = time.time()
        if self.lastframe+self.frame_time <= nowt:
            self.lastframe = time.time()
            self.state += 1
        if self.state >= len(self.frames): self.state = 0

        self.frames[self.state]._process(framebuff)

class ComposedLayer:

    def __init__(
            self,
            init_renderers: list = []
    ):
        self.renderers = init_renderers
    
    def append(self, renderer: object):
        if not hasattr(renderer, "_process"): raise Exception("Render object must have _process attr.")
        self.renderers.append(renderer)
    
    def pop(self, index: int = -1):
        return self.renderers.pop(index)

    def _process(self, framebuff: FrameBuffer):
        for renderer in self.renderers:
            renderer._process(framebuff)

class AppBase:

    SPECSYMBOLS = {
        343 : "KEY_ENTER",
        361 : "KEY_ESCAPE",
        259 : "KEY_UP",
        258 : "KEY_DOWN",
        260 : "KEY_LEFT",
        261 : "KEY_RIGHT",
        263 : "KEY_BACKSPACE",
    }

    def __init__(
            self,
            fps: int | None = None,
            color_mask_max: Color = Color(255, 255, 255),
            color_mask_min: Color = Color(0, 0, 0),
            default_color: Color = Color(255, 255, 255),
            default_color_bg: Color = Color(0, 0, 0),
            nproc: int = 4,
            inputtime: float = 0.0001
    ):
        self.fps = fps
        self.render_queue = []
        self.pre_renders = []
        self.post_renders = []
        self.on_size_change_cbs = []
        self.keyevents = {}
        self.key = ("", True)
        self.terminal = Terminal()
        self.buffer = FrameBuffer(
            self.screen_width,
            self.screen_height,
            color_mask_max,
            color_mask_min,
            default_color,
            default_color_bg
        )

        self.running = False
        self.thread = Thread(target=self.loop)
        self.nproc = nproc
        self.inputtime = inputtime
        self._frame = {}
    
    @property
    def default_color(self): return self.buffer.color_default

    @property
    def default_color_bg(self): return self.buffer.color_default_bg

    @property
    def screen_width(self): return self.terminal.width

    @property
    def screen_height(self): return self.terminal.height

    @property
    def nproc_step(self): return self.buffer.height // self.nproc

    @lru_cache
    def getc(self, r, g, b):
        return self.terminal.color_rgb(r, g, b)

    @lru_cache
    def getcon(self, r, g, b):
        return self.terminal.on_color_rgb(r, g, b)

    def renderer(self, renderer: object):
        if not hasattr(renderer, "_process"): raise Exception("Render object must have _process attr.")
        self.render_queue.append(renderer)
    
    def add_prerender(self, pre_render: Callable, *args, **kwargs):
        self.pre_renders.append((pre_render, args, kwargs), *args, **kwargs)
    
    def add_postrender(self, post_render: Callable, *args, **kwargs):
        self.post_renders.append((post_render, args, kwargs))
    
    def pre_render(self, func: Callable, *args, **kwargs): return self.add_prerender(func, *args, **kwargs)
    def post_render(self, func: Callable, *args, **kwargs): return self.add_postrender(func, *args, **kwargs)
    
    def add_onsizechange(self, func: Callable, *args, **kwargs):
        self.on_size_change_cbs.append((func, args, kwargs))

    def on_size_change(self, func: Callable, *args, **kwargs): return self.add_onsizechange(func, *args, **kwargs)

    def add_keyevent(self, func: Callable, keyevents: tuple, *args, **kwargs):
        self.keyevents[keyevents] = (func, args, kwargs)
    
    def on_key(self, keyevents: tuple, *args, **kwargs):
        def wrapper(func):
            self.add_keyevent(func, keyevents, *args, **kwargs)
        return wrapper

    def _build_chunk(self, ystart: int, yend: int):
        strbuf = io.StringIO()
        for y in range(ystart, yend):
            strbuf.write(f"\033[{y+1};1H")
            last_fg = last_bg = None
            for x in range(self.buffer.width):
                pix = self.buffer.buffer[y * self.buffer.width + x]
                fg, bg = pix.color, pix.color_bg
                if fg != last_fg:
                    strbuf.write(self.getc(*fg))
                    last_fg = fg
                if bg != last_bg:
                    strbuf.write(self.getcon(*bg))
                    last_bg = bg
                strbuf.write(pix.char)
        self._frame[ystart] = strbuf.getvalue()

    def render(self):
        
        for i in range(self.nproc):
            chunk = (
                i*self.nproc_step,
                (i+1)*self.nproc_step
                if i < self.nproc-1
                else self.buffer.height
            )
            Thread(target=self._build_chunk, args=chunk).start()
        framekeys = list(self._frame.keys())
        while len(framekeys) < self.nproc:
            framekeys = list(self._frame.keys())
        framekeys = list(self._frame.keys())
        framevals = list(self._frame.values())

        frame = sorted(framevals, key=lambda x: framekeys[framevals.index(x)])

        sys.stdout.write("".join(frame))
        self._frame.clear()
        sys.stdout.flush()
    
    def get_inputs(self):
        i = self.terminal.inkey(self.inputtime)
        if i.is_sequence:
            if i.code in AppBase.SPECSYMBOLS: self.key = (AppBase.SPECSYMBOLS[i.code], False)
            else: self.key = (i[2], False)
        elif str(i) != "": self.key = (str(i), False)
    
    def get_key(self):
        if self.key[1]: return
        return self.key[0]

    def process_keyevents(self):
        if self.key[1]: return
        for keys, x in self.keyevents.items():
            if self.key[0] in keys:
                x[0](*x[1], **x[2])
                self.key = (self.key[0], True)

    def tick(self):
        if self.screen_width != self.buffer.width or self.screen_height != self.buffer.height:
            for x in self.on_size_change_cbs: x[0](*x[1], **x[2])
            self.buffer.changewh(self.screen_width, self.screen_height)
        
        self.get_inputs()
        self.process_keyevents()
        for renderer in self.render_queue:
            renderer._process(self.buffer)
    
    def loop(self):
        while self.running:
            with self.terminal.cbreak(), self.terminal.hidden_cursor():
                self.tick()

                for x in self.pre_renders: x[0](*x[1], **x[2])
                self.render()
                for x in self.post_renders: x[0](*x[1], **x[2])

                self.buffer.clean()
    
    def run(self):
        self.running = True
        if self.thread is None: self.thread = Thread(target=self.loop)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread = None