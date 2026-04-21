from av.audio.resampler import AudioResampler
from av.video.frame import VideoFrame
from typing import Literal, Iterator
from fractions import Fraction
from av.error import EOFError
from _cffi_backend import FFI
from threading import Thread
import pyray as rl
import numpy as np
import subprocess
import shutil
import json
import time
import av
import os

import tkinter as tk
from tkinter import filedialog

def init_tk():
    root = tk.Tk()
    root.withdraw()

    # Put the (hidden) root on the desired monitor.
    # Using a 1x1 geometry avoids a visible window flash.
    root.geometry(f"1x1+1+1")
    root.update_idletasks()

    return root

def open_file(
    title="Select a file",
    initialdir=None,
    filetypes=(("All files", "*.*"),),
):
    if shutil.which("zenity") is not None:
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=" + title,
                "--file-filter=" + filetypes[0][0] + " | " + filetypes[0][1],
            ],
            capture_output=True,
            text=True
        )

        return result.stdout.strip() or None

    root = init_tk()

    try:
        path = filedialog.askopenfilename(
            title=title,
            initialdir=initialdir,
            filetypes=filetypes,
        )
        return path or None
    finally:
        root.destroy()

def open_file_to_save(
    title="Save as",
    initialdir=None,
    filetypes=(("All files", "*.*"),),
):
    if shutil.which("zenity") is not None:
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--save",
                "--title=" + title,
                "--file-filter=" + filetypes[0][0] + " | " + filetypes[0][1],
            ],
            capture_output=True,
            text=True
        )

        return result.stdout.strip() or None

    root = init_tk()

    try:
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initialdir,
            filetypes=filetypes,
        )
        return path or None
    finally:
        root.destroy()

def position_collides_with_rec(pos: rl.Vector2, rec: rl.Rectangle):
    return pos.x >= rec.x and pos.x <= rec.x + rec.width and pos.y >= rec.y and pos.y <= rec.y + rec.height

def align_to_fps(time: float, fps: float) -> float:
    return int(time * fps) / fps

class Video:
    def __init__(self, file_path: str):
        self.container = av.open(file_path)
        self.stream = self.container.streams.video[0]
        self.duration = self.container.duration / av.time_base
        self.fps = self.stream.average_rate
        self.pos = 0
        self.end_frame = None

    def seek(self, frame_number: int):
        if self.pos == frame_number-1:
            return

        if self.end_frame and frame_number >= self.end_frame:
            return

        frame_time = frame_number * (1 / self.fps)

        self.container.seek(int(frame_time / self.stream.time_base), stream=self.stream)

    def get_frame_at(self, frame_number: int) -> VideoFrame | None:
        frame_time = frame_number * (1 / self.fps)

        self.seek(frame_number)

        try:
            for frame in self.container.decode(self.stream):
                #print(f"Decoding video at {frame.time}")
                if frame.time >= frame_time:
                    self.pos = frame_number
                    return frame
        except EOFError:
            self.end_frame = frame_number
            return None

        print(f"ERROR: Could not find frame {frame_number}")
        return None

    def __del__(self):
        self.container.close()

class Audio:
    def __init__(self, file_path: str):
        self.container = av.open(file_path)
        self.stream = self.container.streams.audio[0]
        self.duration = self.container.duration / av.time_base
        self.fps = self.stream.sample_rate
        self.pos = 0

        self.resampler = AudioResampler(
            format="s16",
            layout="stereo",
            rate=48000,
        )

    def seek(self, frame_number: int):
        if self.pos == frame_number:
            return

        frame_time = frame_number * (1 / self.fps)

        if len(self.container.streams.video) > 0:
            # Use video stream for faster seeking, if it exists.
            # TODO: Maybe we don't need two separate containers?
            video_stream = self.container.streams.video[0]
            self.container.seek(int(frame_time / video_stream.time_base), stream=video_stream)
        else:
            self.container.seek(int(frame_time / self.stream.time_base), stream=self.stream)

    def get_chunk_at(self, chunk_number: int, framesize: int) -> bytes | None:
        frame_number = chunk_number * framesize
        frame_time = frame_number * (1 / self.fps)

        self.seek(frame_number)

        chunk = bytearray()
        chunk_size = 0
        try:
            for frame in self.container.decode(self.stream):
                #print(f"Decoding audio at {frame.time}")
                if frame.time >= frame_time:
                    frame = self.resampler.resample(frame)[0]
                    chunk.extend(frame.to_ndarray().tobytes())
                    chunk_size += frame.samples
                if chunk_size >= framesize:
                    break
        except EOFError:
            return None

        self.pos = frame_number + framesize

        if len(chunk) == 0:
            return None

        return bytes(chunk)

    def __del__(self):
        self.container.close()

class Clip:
    def __init__(self, project: "Project", file_path: str, type: Literal["audio", "video"]):
        self.project = project
        self.file_path = file_path
        self.type = type
        self.video = None
        self.audio = None

        self.audio_graph: dict[str, float] = {}
        self.audio_graph_progress = None
        self.audio_graph_resolution = 30
        self.total_audio_frames = 0

        if self.type == "audio":
            self.audio = Audio(file_path)
        else:
            self.video = Video(file_path)
            if len(self.video.container.streams.audio) > 0:
                self.audio = Audio(file_path)

            self.thumbnail = self.video.get_frame_at(0)
            self.thumbnail_size = None
            self.thumbnail_texture = None

        self.duration = 0
        if self.video:
            self.duration = self.video.duration
        if self.audio and self.audio.duration > self.duration:
            self.duration = self.audio.duration

    def generate_audio_graph(self):
        audio = Audio(self.file_path)

        self.total_audio_frames = audio.duration * audio.fps / audio.stream.frame_size

        frame_count = 0
        last_index = -1

        for frame in audio.container.decode(audio.stream):
            frame_count += 1
            self.audio_graph_progress = frame_count / self.total_audio_frames

            if frame.time is not None:
                frame_index = int(frame.time * self.audio_graph_resolution)
                if frame_index > last_index:
                    last_index = frame_index
                    samples = frame.to_ndarray()

                    if samples.ndim > 1:
                        samples = samples.mean(axis=0)

                    # Normalize if integer format (e.g., int16)
                    if np.issubdtype(samples.dtype, np.integer):
                        max_val = np.iinfo(samples.dtype).max
                        samples = samples / max_val

                    self.audio_graph[str(frame_index)] = float(np.sqrt(np.mean(samples ** 2)))

        self.audio_graph_progress = 1

    def get_thumbnail(self, width, height):
        if self.thumbnail_texture is not None and self.thumbnail_size == (width, height):
            return self.thumbnail_texture
        else:
            if self.thumbnail_texture is not None:
                rl.unload_texture(self.thumbnail_texture)
            thumbnail_texture = create_blank_texture(width, height)
            thumbnail_pixels = frame_to_pixels(self.thumbnail, width, height)
            rl.update_texture(thumbnail_texture, thumbnail_pixels)
            self.thumbnail_texture = thumbnail_texture
            self.thumbnail_size = (width, height)
            return self.thumbnail_texture

    def __del__(self):
        rl.unload_texture(self.thumbnail_texture)

class ClipBinItem:
    def __init__(self, clip: Clip, x: int, y: int):
        self.clip = clip
        self.height = 100
        self.dragging = False
        self.drag_x = 0
        self.drag_y = 0
        self.drag_start = None
        self.x = x
        self.y = y

    def render(self, width: int, left_mouse_down: bool, mouse_pos: rl.Vector2, hover_clip_bin: bool):
        padding = 10
        font_size = 25
        bg_color = rl.Color(69, 69, 69, 255)

        project = self.clip.project

        rec = rl.Rectangle(self.x, self.y, width, self.height)

        if hover_clip_bin and position_collides_with_rec(mouse_pos, rec):
            bg_color = rl.Color(90, 90, 90, 255)

            if left_mouse_down and not self.dragging and not project.dragging:
                self.dragging = True
                project.dragging = True
                self.drag_start = mouse_pos

        if not left_mouse_down:
            if self.dragging:
                self.dragging = False
                project.dragging = False

                # TODO: figure out how to add audio track properly

                group = self.clip.project.group_id
                self.clip.project.group_id += 1

                timeline = self.clip.project.timeline

                if not position_collides_with_rec(mouse_pos, rl.Rectangle(timeline.x, timeline.y, timeline.width, timeline.height)):
                    return

                mouse_time = timeline.get_mouse_time(mouse_pos)

                clips = timeline.get_clips_between(mouse_time, mouse_time + self.clip.duration)

                track_used = False
                track_number = 0
                for i in range(100):
                    track_used = False
                    for clip in clips:
                        if clip.track_number == i:
                            track_used = True
                            break
                    if not track_used:
                        track_number = i
                        break

                if track_used:
                    raise RuntimeError("Too many tracks!")

                timeline.clips.append(TimelineClip(
                    clip=self.clip,
                    type="video",
                    start_time=mouse_time,
                    track_number=track_number+0,
                    group=group,
                ))

                timeline.clips.append(TimelineClip(
                    clip=self.clip,
                    type="audio",
                    start_time=mouse_time,
                    track_number=track_number+1,
                    group=group,
                ))

                timeline.get_end_time()

        if self.dragging:
            rl.end_scissor_mode()
            self.drag_x = mouse_pos.x - self.drag_start.x
            self.drag_y = mouse_pos.y - self.drag_start.y
        else:
            self.drag_x = 0
            self.drag_y = 0

        rec = rl.Rectangle(self.x + self.drag_x, self.y + self.drag_y, width, self.height)

        rl.draw_rectangle_rec(rec, bg_color)
        x_with_pad = int(self.x + self.drag_x + padding)
        y_with_pad = int(self.y + self.drag_y + padding)

        image_width = int(width / 6)
        image_height = self.height - padding * 2
        thumbnail_texture = self.clip.get_thumbnail(image_width, image_height)
        rl.draw_texture(thumbnail_texture, x_with_pad, y_with_pad, rl.WHITE)

        rl.draw_text(os.path.basename(self.clip.file_path), x_with_pad + image_width + padding, y_with_pad, font_size, rl.WHITE)

        if self.clip.audio_graph_progress is not None and self.clip.audio_graph_progress < 1:
            progress_bar_height = 5
            progress_bar_width = width - padding * 3 - image_width
            rl.draw_rectangle(x_with_pad + image_width + padding, y_with_pad + image_height - padding - progress_bar_height, progress_bar_width, progress_bar_height, rl.Color(30, 30, 30, 255))
            rl.draw_rectangle(x_with_pad + image_width + padding, y_with_pad + image_height - padding - progress_bar_height, int(progress_bar_width * self.clip.audio_graph_progress), progress_bar_height, rl.Color(200, 200, 200, 255))

class ClipBin:
    def __init__(self, x: int, y: int, ):
        self.clips: list[ClipBinItem] = []
        self.scroll = 0
        self.x = x
        self.y = y

    def add_clip(self, clip: Clip):
        self.clips.append(ClipBinItem(clip, self.x, self.y))

    def render(self, width: int, height: int, left_mouse_down: bool, mouse_pos: rl.Vector2, mouse_wheel_move: float, delta_time: float):
        pos_y = int(self.y + self.scroll)
        margin = 5

        rec = rl.Rectangle(self.x, self.y, width+1, height+1)
        hover_clip_bin = position_collides_with_rec(mouse_pos, rec)

        rl.draw_rectangle_lines_ex(rec, 1, rl.WHITE)

        render_order: list[ClipBinItem] = []

        for item in self.clips:
            item.y = pos_y
            if item.dragging:
                render_order.append(item)
            else:
                render_order.insert(0, item)

            pos_y += item.height + margin

        for item in render_order:
            rl.begin_scissor_mode(self.x, self.y, width, height)
            item.render(width, left_mouse_down, mouse_pos, hover_clip_bin)
            rl.end_scissor_mode()

        max_scroll = min(0, height - (pos_y - self.scroll - self.y))

        if position_collides_with_rec(mouse_pos, rl.Rectangle(self.x, self.y, width, height)):
            if mouse_wheel_move != 0:
                scroll_amount = mouse_wheel_move * delta_time * 800
                self.scroll += scroll_amount

                if self.scroll > 0:
                    self.scroll = 0
                if self.scroll < max_scroll:
                    self.scroll = max_scroll

class Message:
    def __init__(self, text: str):
        self.text = text
        self.time = time.time()

class Project:
    def __init__(self, fps: int, audio_fps: int):
        self.file_path = None

        self.timeline = Timeline(audio_fps)
        self.group_id = 0
        self.clip_bin = ClipBin(0, 0)
        self.fps = fps
        self.audio_fps = audio_fps
        self.dragging = False
        self.rendering = False
        self.render_file: str | None = None
        self.render_progress: float = 0.0

        self.mode: Literal["cut", "select", "move"] = "select"

        self.history = []
        self.history_index = None

        self.messages: list[Message] = []

        # TODO: redoing load project screws up timeline
        #self.save_history("New project")

    def show_message(self, message: str):
        print(f"Message: {message}")
        self.messages.append(Message(message))
        if len(self.messages) > 1:
            self.messages.pop(0)

    def save_history(self, action_name: str):
        print(f"Saving history '{action_name}'")

        if self.history_index is not None:
            self.history = self.history[:self.history_index+1]

        self.history.append({
            "name": action_name,
            "state": self.make_project_json(),
        })

        if len(self.history) > 100:
            self.history.pop(0)

        self.history_index = len(self.history)-1

    def undo_history(self):
        if self.history_index is None:
            self.show_message("INFO: Nothing to undo")
            return

        try:
            current = self.history[self.history_index]
            self.history_index -= 1
            if self.history_index < 0:
                raise IndexError()
            previous = self.history[self.history_index]
            self.load_json(previous["state"])
            self.show_message(f"INFO: Undo '{current['name']}' -> '{previous['name']}'")
        except IndexError:
            self.show_message("INFO: Nothing to undo")
            self.history_index = 0

    def redo_history(self):
        self.history_index += 1

        try:
            history = self.history[self.history_index]
            self.load_json(history["state"])
            self.show_message(f"INFO: Redo '{history['name']}'")
        except IndexError:
            self.show_message("INFO: Nothing to redo")
            self.history_index = len(self.history)-1

    def build_video_buffer(self) -> bool:
        if self.rendering:
            return False

        buf_secs = 3.0
        ph = self.timeline.playhead

        buffer_start = int((ph - buf_secs) * self.fps)

        frame_found = False
        for frame in range(int(ph * self.fps), int((ph + buf_secs) * self.fps)):
            time = frame / self.fps + 0.000001

            if frame not in self.timeline.video_buffer:
                video_frame = self.timeline.get_video_frame_at(time)
                self.timeline.video_buffer[frame] = video_frame
                frame_found = True
                break

        for key in list(self.timeline.video_buffer.keys()):
            if key < buffer_start or key > frame:
                try:
                    del self.timeline.video_buffer[key]
                except KeyError:
                    print(f"WARNING: Key {key} not found in video buffer")

        return not frame_found

    def build_audio_buffer(self) -> bool:
        if self.rendering:
            return False

        buf_secs = 3.0
        ph = self.timeline.playhead

        buffer_start = int((ph - buf_secs) * self.audio_fps) // self.timeline.audio_framesize
        current_chunk = int(int(ph * self.audio_fps) // self.timeline.audio_framesize)
        last_chunk = int((ph + buf_secs) * self.audio_fps) // self.timeline.audio_framesize

        frame_found = False
        for chunk_number in range(current_chunk, last_chunk):
            frame = chunk_number * self.timeline.audio_framesize
            time = frame / self.audio_fps + 0.000001

            if chunk_number not in self.timeline.audio_buffer:
                chunk = self.timeline.get_audio_chunk_at(time)
                self.timeline.audio_buffer[chunk_number] = chunk
                frame_found = True
                break

        for key in list(self.timeline.audio_buffer.keys()):
            if key < buffer_start or key > chunk_number:
                try:
                    del self.timeline.audio_buffer[key]
                except KeyError:
                    print(f"WARNING: Key {key} not found in audio buffer")

        return not frame_found

    def make_project_json(self):
        project_json = {
            "fps": self.fps,
            "audio_fps": self.audio_fps,
            "clips": [],
            "timeline": [],
        }

        clip_files = {}

        for index, clip in enumerate(self.clip_bin.clips):
            clip_json = {
                "file_path": clip.clip.file_path,
                "type": clip.clip.type,
            }

            if clip.clip.audio_graph_progress == 1:
                clip_json["audio_graph"] = clip.clip.audio_graph
                clip_json["audio_graph_resolution"] = clip.clip.audio_graph_resolution
                clip_json["total_audio_frames"] = clip.clip.total_audio_frames

            project_json["clips"].append(clip_json)
            clip_files[clip.clip.file_path] = index

        for clip in self.timeline.clips:
            project_json["timeline"].append({
                "track_number": clip.track_number,
                "clip": clip_files[clip.clip.file_path],
                "type": clip.type,  # TODO: duplicate reference?
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "clip_start": clip.clip_start,
                "group": clip.group,
            })

        return project_json

    def save(self, file_path: str):
        project_json = self.make_project_json()

        with open(file_path, "w") as f:
            json.dump(project_json, f, indent=4)

        self.show_message(f"Saved project to {file_path}")

        self.file_path = file_path

    def load_json(self, project_json: dict, append: bool = False):
        self.fps = project_json.get("fps", 30)
        self.audio_fps = project_json.get("audio_fps", 48000)

        clips = project_json.get("clips")

        if clips is None:
            self.show_message(f"ERROR: Project file does not contain clips!")
            return

        old_clips = self.clip_bin.clips

        if append:
            offset = self.timeline.get_end_time()
        else:
            offset = 0
            self.timeline.clips = []
            self.clip_bin.clips = []
            self.group_id = 1

        clip_by_index = {}

        for index, clip in enumerate(clips):
            clip_file_path = clip.get("file_path")
            if not clip_file_path:
                self.show_message(f"ERROR: Clip {index} does not have a file path!")
                return

            if not os.path.exists(clip_file_path):
                self.show_message(f"ERROR: Clip '{clip_file_path}' does not exist!")
                return

            clip_type = clip.get("type")
            if not clip_type:
                self.show_message(f"ERROR: Clip {index} does not have a type!")
                return

            # Reuse existing clips with same file path
            # to avoid having to regenerate audio graph
            clip_obj = None
            for c in old_clips:
                if c.clip.file_path == clip_file_path:
                    clip_obj = c.clip
                    break

            if clip_obj is None:
                clip_obj = Clip(self, clip_file_path, clip_type)

                audio_graph = clip.get("audio_graph")
                audio_graph_resolution = clip.get("audio_graph_resolution")
                total_audio_frames = clip.get("total_audio_frames")
                if audio_graph:
                    clip_obj.audio_graph_progress = 1
                    clip_obj.audio_graph = audio_graph
                    clip_obj.audio_graph_resolution = audio_graph_resolution
                    clip_obj.total_audio_frames = total_audio_frames

                self.clip_bin.add_clip(clip_obj)
            elif not append:
                self.clip_bin.add_clip(clip_obj)

            clip_by_index[index] = clip_obj

            print(f"Added clip {index}")

        timeline = project_json.get("timeline")
        next_group = self.group_id

        if timeline is None:
            self.show_message(f"ERROR: Project file does not contain a timeline!")
            return

        for index, clip in enumerate(timeline):
            clip_index = clip.get("clip")
            if clip_index is None:
                self.show_message(f"ERROR: Timeline clip {index} does not have a clip!")
                return

            type = clip.get("type")
            if type is None:
                self.show_message(f"ERROR: Timeline clip {index} does not have a type!")
                return

            start_time = clip.get("start_time")
            if start_time is None:
                self.show_message(f"ERROR: Timeline clip {index} does not have a start time!")
                return

            end_time = clip.get("end_time")
            if end_time is None:
                self.show_message(f"ERROR: Timeline clip {index} does not have a end time!")
                return

            track_number = clip.get("track_number")
            if track_number is None:
                self.show_message(f"ERROR: Timeline clip {index} does not have a track number!")
                return

            group = clip.get("group")
            if group is not None:
                group += next_group
                if group >= self.group_id:
                    self.group_id = group + 1

            self.timeline.clips.append(TimelineClip(
                clip=clip_by_index[clip_index],
                type=type,
                start_time=start_time+offset,
                end_time=end_time+offset,
                track_number=track_number,
                clip_start=clip.get("clip_start", 0),
                group=group,
            ))

            print(f"Added timeline clip {index} (offset {offset})")

        self.timeline.get_end_time()

    def load(self, file_path: str, append: bool = False) -> None:
        try:
            with open(file_path, "r") as f:
                project_json = json.load(f)
        except OSError as e:
            self.show_message(f"ERROR: Could not load file '{file_path}': {e}")
            return
        except json.JSONDecodeError:
            self.show_message(f"ERROR: Invalid project file '{file_path}'")
            return
        except Exception:
            self.show_message(f"ERROR: Invalid project file '{file_path}'")
            return

        self.load_json(project_json, append)

        self.file_path = file_path

    def start_rendering(self) -> None:
        if self.rendering:
            self.show_message("ERROR: Can't render two things at once")
            return

        self.rendering = True

    def render(self) -> None:
        if not self.render_file:
            self.show_message("ERROR: No render file provided")
            self.rendering = False
            return

        self.rendering = True
        self.render_progress = 0.0

        render_start = time.time()

        try:
            end_time = self.timeline.get_end_time()

            with av.open(self.render_file, mode="w") as container:
                frame_count = int(end_time * self.fps)
                audio_frame_count = int(end_time * self.audio_fps)
                chunk_count = int(audio_frame_count / self.timeline.audio_framesize)

                total_steps = frame_count + chunk_count
                progress = 0

                first_frame = self.timeline.get_video_frame_at(0)

                print(f"Rendering {frame_count} frames ({end_time} seconds) in {first_frame.format.name}")

                # Add a video stream
                stream = container.add_stream("libx264", rate=self.fps)
                stream.width = first_frame.width  # TODO: set size of project
                stream.height = first_frame.height
                stream.pix_fmt = first_frame.format.name
                stream.time_base = Fraction(1, self.fps)

                # Add audio stream
                audio_stream = container.add_stream("aac", rate=self.audio_fps)
                audio_stream.time_base = Fraction(1, self.audio_fps)
                channels = 2

                # RENDER VIDEO
                for frame_index in range(frame_count):
                    frame_time = frame_index * (1 / self.fps)
                    frame = self.timeline.get_video_frame_at(frame_time)

                    if frame is None:
                        black = np.zeros((stream.height, stream.width, 3), dtype=np.uint8)
                        frame = av.VideoFrame.from_ndarray(black, format="rgb24")

                    frame = frame.reformat(
                        width=stream.width,
                        height=stream.height,
                        format="rgba",
                    )

                    frame = av.VideoFrame.from_bytes(
                        bytes(frame.planes[0]),
                        frame.width,
                        frame.height,
                        format="rgba",
                    )

                    # Encode + mux
                    for packet in stream.encode(frame):
                        container.mux(packet)

                    progress += 1
                    self.render_progress = progress / total_steps

                # Flush encoder
                for packet in stream.encode(None):
                    container.mux(packet)

                # RENDER AUDIO
                audio_pts = 0
                for chunk_index in range(chunk_count):
                    chunk_time = chunk_index * self.timeline.audio_framesize * (1 / self.audio_fps) + 0.000001
                    chunk = self.timeline.get_audio_chunk_at(chunk_time)

                    if chunk is None:
                        chunk = bytearray(self.timeline.audio_framesize*4)
                    else:
                        print(f"Audio chunk len: {len(chunk)}")
                        chunk = chunk + bytearray(self.timeline.audio_framesize*4 - len(chunk))

                    nd_frame = np.frombuffer(chunk, dtype=np.int16)

                    # reshape to (num_samples, channels) for interleaved input
                    if nd_frame.size % channels != 0:
                        # Drop trailing partial sample
                        nd_frame = nd_frame[: nd_frame.size - (nd_frame.size % channels)]
                    nd_frame = nd_frame.reshape(1, -1)

                    frame = av.AudioFrame.from_ndarray(nd_frame)
                    frame.sample_rate = self.audio_fps
                    frame.time_base = audio_stream.time_base
                    frame.pts = audio_pts

                    audio_pts += frame.samples

                    # Encode + mux
                    for packet in audio_stream.encode(frame):
                        container.mux(packet)

                    progress += 1
                    self.render_progress = progress / total_steps

                # Flush encoder
                for packet in audio_stream.encode(None):
                    container.mux(packet)

            self.show_message(f"Rendering finished in {timestamp(time.time()-render_start)}!")
        finally:
            self.rendering = False
            self.render_file = None

class TimelineClip:
    def __init__(self,
                 track_number: int,
                 clip: Clip,
                 type: Literal["audio", "video"],
                 start_time: float,
                 end_time: float | None = None,
                 clip_start: float = 0.0,
                 group: int | None = None,
    ):
        self.clip = clip
        self.clip_start = clip_start
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.track_number = track_number
        self.group = group
        self.selected = False
        self.last_selected = False
        self.texture = None
        self.texture_updated = 0
        self.texture_scroll_x = 0
        self.drag_start = None
        self.was_dragged = False
        self.pressed_down = False
        self.initiated_move = False
        self.resize_left_start = None
        self.resize_right_start = None

        if self.type == "audio":
            self.resource = clip.audio
        else:
            self.resource = clip.video

        if self.end_time is None:
            self.end_time = start_time + self.resource.duration

    def get_grouped_clips(self) -> list["TimelineClip"]:
        if self.group is None:
            grouped_clips = [self]
        else:
            grouped_clips = []
            for clip in self.clip.project.timeline.clips:
                if clip.group == self.group:
                    grouped_clips.append(clip)
        return grouped_clips

    def get_last_selected_clip(self) -> "TimelineClip | None":
        for clip in self.clip.project.timeline.clips:
            if clip.last_selected:
                return clip
        return None

    def render(self, padding_top: int, left_mouse_pressed: bool, mouse_pos: rl.Vector2, ctrl_down: bool, shift_down: bool, left_mouse_released: bool, left_mouse_down: bool):
        timeline = self.clip.project.timeline

        track_height = 70
        track_padding = 5

        clip_x1 = max(0, int((self.start_time - timeline.scroll_x) * timeline.zoom))
        clip_x2 = min(
            int((self.end_time - timeline.scroll_x) * timeline.zoom),
            timeline.x + timeline.width,
        )
        w = clip_x2 - clip_x1

        if w < 1:
            return # drag issue

        color = {0: rl.BLUE, 1: rl.GREEN, 2: rl.ORANGE}[self.track_number % 3]
        clip_y = timeline.y + padding_top + (track_height + track_padding) * self.track_number
        rec = rl.Rectangle(timeline.x + clip_x1, clip_y, w, track_height)
        #rl.draw_rectangle_rec(rec, color)

        #draw_text(timestamp(self.clip_start), int(rec.x)+3, int(rec.y)+3, 12, YELLOW)
        #draw_text(f"{self.clip_start}", int(rec.x)+3, int(rec.y)+3+15, 12, YELLOW)
        #draw_text(f"{self.clip_start*self.clip.video.fps}", int(rec.x)+3, int(rec.y)+3+15+15, 12, YELLOW)

        if self.type == "audio" and w > 10:
            if self.texture_updated < time.time() - 1 or self.texture.width != w or self.texture.height != track_height or self.texture_scroll_x != timeline.scroll_x:
                audio_resolution = 1
                cur_time = max(0, timeline.scroll_x - self.start_time) + self.clip_start

                if self.clip.total_audio_frames == 0:
                    return

                volume = 0
                graph = bytearray(w*4*track_height)
                for x in range(0, w, audio_resolution):
                    cur_time += audio_resolution / timeline.zoom
                    index = str(int(cur_time * self.clip.audio_graph_resolution))
                    vol = self.clip.audio_graph.get(index)
                    if vol is not None:
                        volume = min(track_height, int(vol * 2 * track_height))

                    for y in range(track_height - volume, track_height):
                        graph[w*4*y+x*4+0] = 255
                        graph[w*4*y+x*4+1] = 255
                        graph[w*4*y+x*4+2] = 255
                        graph[w*4*y+x*4+3] = 255

                if self.texture is None or self.texture.width != w or self.texture.height != track_height:
                    if self.texture is not None:
                        rl.unload_texture(self.texture)
                    self.texture = create_blank_texture(w, track_height)

                rl.update_texture(self.texture, rl.ffi.new("char[]", bytes(graph)))
                self.texture_updated = time.time()
                self.texture_scroll_x = timeline.scroll_x

        if self.selected:
            line_color = rl.ORANGE
        else:
            line_color = rl.Color(0, 0, 0, 100)

        line_width = 2

        if rec.width > 5:
            rl.draw_rectangle_rec(rec, color)
        else:
            color = (color[0], color[1], color[2], 145)
            rl.draw_rectangle_rec(rec, color)

        if self.texture:
            rl.draw_texture(self.texture, int(rec.x), int(rec.y) - line_width, rl.WHITE)

        if rec.width > 5:
            rl.draw_rectangle_lines_ex(rec, line_width, line_color)

        if w <= 5:
            return

        resize_left_rec = rl.Rectangle(rec.x + 1, rec.y, 10, rec.height)
        resize_right_rec = rl.Rectangle(rec.x + rec.width - 10, rec.y, 10, rec.height)

        if position_collides_with_rec(mouse_pos, resize_left_rec):
            rl.draw_rectangle_rec(resize_left_rec, rl.ORANGE)

            if left_mouse_pressed:
                if self.resize_left_start is None and not self.clip.project.dragging:
                    self.resize_left_start = mouse_pos
                    self.clip.project.dragging = True

        elif position_collides_with_rec(mouse_pos, resize_right_rec):
            rl.draw_rectangle_rec(resize_right_rec, rl.ORANGE)

            if left_mouse_pressed:
                if self.resize_right_start is None and not self.clip.project.dragging:
                    self.resize_right_start = mouse_pos
                    self.clip.project.dragging = True

        if position_collides_with_rec(mouse_pos, rec):
            if left_mouse_pressed:
                # Cut clip
                if self.clip.project.mode == "cut" and not self.clip.project.dragging:
                    abs_time = timeline.get_mouse_time(mouse_pos)

                    affected_clips = self.get_grouped_clips()

                    new_group_id = self.clip.project.group_id
                    self.clip.project.group_id += 1

                    for clip in affected_clips:
                        rel_time = abs_time - clip.start_time + clip.clip_start
                        end_time = clip.end_time
                        clip.end_time = abs_time

                        timeline.clips.append(TimelineClip(
                            track_number=clip.track_number,
                            clip=clip.clip,
                            type=clip.type,
                            start_time=abs_time,
                            clip_start=rel_time,
                            end_time=end_time,
                            group=new_group_id,
                        ))

                        timeline.get_end_time()

                    self.clip.project.save_history("Cut clip")
                elif self.clip.project.mode == "select":
                    if not ctrl_down and not self.clip.project.dragging and self.drag_start is None:
                        self.drag_start = mouse_pos
                        self.clip.project.dragging = True
                        self.was_dragged = False

                    if not self.selected or ctrl_down:
                        affected_clips = None
                        if shift_down:
                            last_selected = self.get_last_selected_clip()
                            if last_selected:
                                if last_selected.end_time > self.end_time:
                                    affected_clips = timeline.get_clips_between(self.start_time, last_selected.end_time)
                                else:
                                    affected_clips = timeline.get_clips_between(last_selected.start_time, self.end_time)

                        if affected_clips is None:
                            affected_clips = self.get_grouped_clips()

                        if not ctrl_down:
                            for clip in timeline.clips:
                                clip.selected = False

                        for clip in affected_clips:
                            clip.selected = not clip.selected
                    elif self.selected:
                        self.pressed_down = True
                elif self.clip.project.mode == "move" and not self.clip.project.dragging:
                    if not self.clip.project.dragging and self.drag_start is None:
                        self.drag_start = mouse_pos
                        self.clip.project.dragging = True
                        self.was_dragged = False
                        self.initiated_move = True

                    affected_clips = timeline.get_clips_between(self.start_time, timeline.end_time)

                    for clip in timeline.clips:
                        clip.selected = False

                    for clip in affected_clips:
                        clip.selected = True

                if self.selected:
                    for clip in timeline.clips:
                        clip.last_selected = False
                    self.last_selected = True

        if self.resize_left_start is not None:
            diff_x = mouse_pos.x - self.resize_left_start.x

            affected_clips = self.get_grouped_clips()

            for clip in affected_clips:
                new_start = clip.start_time + diff_x / timeline.zoom

                max_start = clip.end_time - 1 / clip.clip.project.fps
                min_start = max(0, clip.start_time - clip.clip_start)

                if new_start > max_start:
                    new_start = max_start
                elif new_start < min_start:
                    new_start = min_start

                last_time = 0
                for c in timeline.get_clips_between(new_start, clip.start_time):
                    if c != clip and c.track_number == clip.track_number:
                        if c.end_time > last_time and c.end_time <= clip.start_time:
                            last_time = c.end_time
                new_start = max(last_time, new_start)

                diff = new_start - clip.start_time

                clip.start_time = new_start
                clip.clip_start += diff

                self.resize_left_start = mouse_pos
                self.clip.project.timeline.video_buffer = {}
                self.clip.project.timeline.audio_buffer = {}
                self.clip.project.timeline.get_end_time()
        elif self.resize_right_start is not None:
            diff_x = mouse_pos.x - self.resize_right_start.x

            affected_clips = self.get_grouped_clips()

            for clip in affected_clips:
                new_end = clip.end_time + diff_x / timeline.zoom

                max_end = clip.clip.duration - clip.clip_start + clip.start_time
                min_end = clip.start_time + 1 / clip.clip.project.fps

                if new_end > max_end:
                    new_end = max_end
                elif new_end < min_end:
                    new_end = min_end

                first_time = None
                for c in timeline.get_clips_between(clip.end_time, new_end):
                    if c != clip and c.track_number == clip.track_number:
                        if (first_time is None or c.start_time < first_time) and c.start_time >= clip.start_time:
                            first_time = c.start_time

                if first_time is not None:
                    new_end = min(first_time, new_end)

                clip.end_time = new_end

                self.resize_right_start = mouse_pos
                self.clip.project.timeline.video_buffer = {}
                self.clip.project.timeline.audio_buffer = {}
                self.clip.project.timeline.get_end_time()
        elif self.drag_start is not None:
            diff_x = mouse_pos.x - self.drag_start.x

            if self.clip.project.mode == "move":
                start_time = None
                end_time = None
                for clip in timeline.clips:
                    if clip.selected:
                        if start_time is None or start_time > clip.start_time:
                            start_time = clip.start_time
                        if end_time is None or end_time < clip.end_time:
                            end_time = clip.end_time
                affected_clips = [
                    TimelineClip(
                        track_number=clip.track_number,
                        clip=clip.clip,
                        type="virtual",
                        start_time=start_time,
                        end_time=end_time,
                        group=clip.group,
                    )
                ]
            else:
                affected_clips = self.clip.project.timeline.get_selected_clips()

            offset = None

            for clip in affected_clips:
                if diff_x < 0:
                    self.was_dragged = True
                    new_start = clip.start_time + diff_x / timeline.zoom
                    last_time = 0
                    for c in timeline.get_clips_between(new_start, clip.end_time - (clip.start_time - new_start)):
                        if c not in affected_clips and c.track_number == clip.track_number:
                            if c.end_time > last_time and c.end_time <= clip.start_time:
                                last_time = c.end_time
                    new_start = max(last_time, new_start)

                    diff = -(clip.start_time - new_start)
                    new_end = clip.end_time + diff

                    if offset is None or offset < diff or clip.start_time + offset < 0:
                        offset = diff
                elif diff_x > 0:
                    self.was_dragged = True
                    new_end = clip.end_time + diff_x / timeline.zoom
                    first_time = None
                    for c in timeline.get_clips_between(clip.start_time + (new_end - clip.end_time), new_end):
                        if c not in affected_clips and c.track_number == clip.track_number:
                            if (first_time is None or c.start_time < first_time) and c.start_time >= clip.end_time:
                                first_time = c.start_time

                    if first_time is not None:
                        new_end = min(first_time, new_end)

                    diff = (new_end - clip.end_time)
                    new_start = clip.start_time + diff

                    if offset is None or offset > diff:
                        offset = diff

            if offset:
                overlap = False
                if self.clip.project.mode == "move":
                    affected_clips = timeline.get_selected_clips()
                else:
                    for clip in affected_clips:
                        new_start = clip.start_time + offset
                        new_end = clip.end_time + offset
                        for c in timeline.get_clips_between(new_start, new_end):
                            if c not in affected_clips and c.track_number == clip.track_number:
                                overlap = True
                                break
                        if overlap:
                            break
                if not overlap:
                    for clip in affected_clips:
                        clip.start_time += offset
                        clip.end_time += offset
                    self.drag_start = mouse_pos
                    self.clip.project.timeline.video_buffer = {}
                    self.clip.project.timeline.audio_buffer = {}
                    self.clip.project.timeline.get_end_time()

            if not left_mouse_down:
                self.clip.project.dragging = False
                self.drag_start = None

        if left_mouse_released:
            if self.resize_left_start:
                self.resize_left_start = None
                self.clip.project.dragging = False
            elif self.resize_right_start:
                self.resize_right_start = None
                self.clip.project.dragging = False

            if self.initiated_move:
                self.initiated_move = False
                affected_clips = self.get_grouped_clips()

                for clip in timeline.clips:
                    clip.selected = False

                for clip in affected_clips:
                    clip.selected = True
            elif self.pressed_down and not self.was_dragged:
                self.pressed_down = False
                if self.selected:
                    for clip in timeline.clips:
                        clip.selected = False

                    affected_clips = self.get_grouped_clips()

                    for clip in affected_clips:
                        clip.selected = True

        return rec, color, line_color

    def __del__(self):
        if self.texture is not None:
            # TODO: We don't unload textures as they go off screen
            # so when we delete or extract a large number of clips
            # they will be unloaded at that time and it is slow
            rl.unload_texture(self.texture)

class Timeline:
    def __init__(self, audio_fps: int):
        self.audio_offset = 0  # 0.17
        self.audio_framesize = 1024
        self.clips: list[TimelineClip] = []
        self.playhead = 0.0
        self.audio_playhead = 0
        self.length = 3600.0
        self.zoom = 100.0
        self.scroll_x = 0.0
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.play_current_audio_frame = False
        self.is_playing = False
        self.buffer_ready = False
        self.want_to_play = False
        self.audio_fps = audio_fps
        self.video_buffer: dict[int, VideoFrame] = {}
        self.audio_buffer: dict[int, bytearray] = {}
        self.end_time: float = 0.0

        self.audio_bufsize = int(self.audio_framesize*4)

        rl.init_audio_device()
        rl.set_audio_stream_buffer_size_default(self.audio_bufsize)

        self.audio_stream = rl.load_audio_stream(self.audio_fps, 16, 2)
        rl.play_audio_stream(self.audio_stream)

    def get_end_time(self) -> float:
        end_time = 0
        for clip in self.clips:
            if clip.end_time > end_time:
                end_time = clip.end_time
        self.end_time = end_time  # Cache end time
        return end_time

    def get_video_frame_at(self, time: float) -> VideoFrame | None:
        clips = self.get_clips_at(time)
        for clip in clips:
            if clip.type == "video":
                clip_time = time - clip.start_time + clip.clip_start
                clip_frame = int(clip_time * clip.resource.fps)
                #print(f"Getting video frame at {time}/{clip_frame}/{clip_time}")
                return clip.resource.get_frame_at(clip_frame)

    def get_audio_chunk_at(self, time: float) -> bytes | None:
        time -= self.audio_offset
        if time < 0:
            time = 0
        clips = self.get_clips_at(time)
        for clip in clips:
            if clip.type == "audio":
                clip_time = time - clip.start_time + clip.clip_start
                clip_frame = clip_time * clip.resource.fps
                clip_chunk_number = int(clip_frame // self.audio_framesize)
                #print(f"Getting audio frame {time}/{clip_frame}/{clip_time}")
                return clip.resource.get_chunk_at(clip_chunk_number, self.audio_framesize)

    def multiple_clips_selected(self) -> bool:
        selected_clip = None
        for clip in self.clips:
            if clip.selected:
                if not selected_clip:
                    selected_clip = clip
                elif selected_clip.group != clip.group:
                    return True
        return False

    def get_mouse_time(self, mouse_pos: rl.Vector2):
        return self.scroll_x + (mouse_pos.x - self.x) * (1 / self.zoom)

    def get_current_clips(self) -> Iterator[TimelineClip]:
        return self.get_clips_at(self.playhead)

    def get_clips_at(self, time: float, track_number: int | None = None) -> list[TimelineClip]:
        clips = []
        for clip in self.clips:
            if clip.end_time >= time and clip.start_time <= time:
                if track_number is None or clip.track_number == track_number:
                    clips.append(clip)
        return clips

    def get_clips_between(self, start_time: float, end_time: float) -> list[TimelineClip]:
        clips = []
        for clip in self.clips:
            if clip.end_time > start_time and clip.start_time < end_time:
                clips.append(clip)
        return clips

    def get_selected_clips(self) -> list[TimelineClip]:
        selected_clips = []
        for clip in self.clips:
            if clip.selected:
                selected_clips.append(clip)
        return selected_clips

    def jump(self, seconds: float, relative: bool = True):
        if self.is_playing:
            self.is_playing = False
            self.buffer_ready = False
            self.want_to_play = True

        if relative:
            self.playhead += seconds
        else:
            self.playhead = seconds

        if self.playhead < 0:
            self.playhead = 0
        elif self.playhead > self.end_time:
            self.playhead = self.end_time

        self.scroll_to_playhead()

        rl.stop_audio_stream(self.audio_stream)

        for clip in self.get_current_clips():
            if clip.type == "audio":
                self.audio_playhead = int(self.playhead * self.audio_fps)
                if not self.is_playing:
                    self.play_current_audio_frame = True
                rl.play_audio_stream(self.audio_stream)
                break

    def jump_to_empty(self):
        if len(self.clips) == 0:
            return

        earliest_empty = self.end_time + 0.01

        for clip in self.clips:
            if clip.start_time <= earliest_empty and clip.end_time >= earliest_empty:
                earliest_empty = self.end_time

            if clip.end_time > self.playhead:
                empty = clip.end_time + 0.01
                if earliest_empty > empty:
                    not_empty = False
                    for c in self.clips:
                        if c.start_time <= empty and c.end_time >= empty:
                            not_empty = True
                            break
                    if not not_empty:
                        earliest_empty = empty

        self.jump(earliest_empty - 0.01, False)

        self.clips[0].clip.project.show_message(f"Jumped to {timestamp(earliest_empty)}")

    def scroll_to_playhead(self):
        self.scroll_x = self.playhead - self.width / 2 / self.zoom
        if self.scroll_x < 0:
            self.scroll_x = 0

    def render(self, x: int, y: int, width: int, height: int, mouse_wheel_move: float, delta_time: float, ctrl_down: bool, shift_down: bool, left_mouse_pressed: bool, mouse_pos: rl.Vector2, left_mouse_released: bool, left_mouse_down: bool):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        minute_line_height = 25
        second5_line_height = 15
        second_line_height = 5
        min_gap = 3

        # TODO: Get project in a sensible way
        if len(self.clips) > 0:
            project = self.clips[0].clip.project
        else:
            project = None

        rl.begin_scissor_mode(x, y, width, height)

        modulo = self.scroll_x % 60
        timeline_start = int(modulo * self.zoom * -1)

        minute_in_pixels = 60 * self.zoom
        second_in_pixels = 1 * self.zoom

        if minute_in_pixels >= min_gap:
            i = timeline_start
            while i <= width:
                rl.draw_line(x + int(i), y, x + int(i), y + minute_line_height, rl.WHITE)
                i += minute_in_pixels

        if second_in_pixels >= min_gap:
            i = timeline_start
            while i <= width:
                rl.draw_line(x + int(i), y, x + int(i), y + second_line_height, rl.WHITE)
                i += second_in_pixels

        if second_in_pixels * 5 >= min_gap:
            i = timeline_start
            while i <= width:
                rl.draw_line(x + int(i), y, x + int(i), y + second5_line_height, rl.WHITE)
                i += second_in_pixels * 5

        clips = self.get_clips_between(self.scroll_x, self.scroll_x + width / self.zoom)
        for clip in clips:
            clip.render(minute_line_height, left_mouse_pressed, mouse_pos, ctrl_down, shift_down, left_mouse_released, left_mouse_down)

        rl.end_scissor_mode()

        if position_collides_with_rec(mouse_pos, rl.Rectangle(x, y, width, height)):
            mouse_time = self.get_mouse_time(mouse_pos)

            if left_mouse_pressed:
                self.playhead = mouse_time
                self.audio_playhead = int(self.playhead * 48000)
                self.play_current_audio_frame = True
                self.is_playing = False

            scroll_amount = mouse_wheel_move * delta_time
            if scroll_amount != 0:
                if ctrl_down:
                    min_zoom = 0.01
                    max_zoom = 2000
                    self.zoom *= 1.2 if scroll_amount > 0 else 0.8
                    if self.zoom < min_zoom:
                        self.zoom = min_zoom
                    elif self.zoom > max_zoom:
                        self.zoom = max_zoom

                    self.scroll_x = mouse_time - (mouse_pos.x - x) * (1 / self.zoom)
                    if self.scroll_x < 0:
                        self.scroll_x = 0
                else:
                    self.scroll_x -= scroll_amount * 4000 / self.zoom

                    if self.scroll_x < 0:
                        self.scroll_x = 0

            # Mouse time line
            rl.draw_line(int(mouse_pos.x), y, int(mouse_pos.x), y + height, rl.WHITE)

            # Draw mouse time
            stamp = timestamp(mouse_time)
        else:
            # Draw playhead time
            stamp = timestamp(self.playhead)

        stamp_size = 25
        stamp_width = rl.measure_text(stamp, stamp_size)
        rl.draw_text(stamp, x + width - stamp_width - 20, y - 35, stamp_size, rl.WHITE)

        # Playhead line
        playhead_x = int((self.playhead - self.scroll_x) * self.zoom)
        rl.draw_line(playhead_x, y, playhead_x, y + height, rl.WHITE)

        if self.is_playing and playhead_x > x + width:
            self.scroll_to_playhead()

        if ctrl_down:
            if shift_down:
                if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
                    selected_clips = self.get_selected_clips()

                    self.extend(selected_clips, (1 / project.fps), 0)

                    if len(selected_clips) == 1:
                        msg = "Extended clip from start"
                    else:
                        msg = f"Extended {len(selected_clips)} clips from start"

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()

                    project.save_history(msg)
                elif rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                    selected_clips = self.get_selected_clips()

                    self.shrink(selected_clips, (1 / project.fps), 0)

                    if len(selected_clips) == 1:
                        msg = "Shrunk clip from start"
                    else:
                        msg = f"Shrunk {len(selected_clips)} clips from start"

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()

                    project.save_history(msg)
            else:
                if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
                    selected_clips = self.get_selected_clips()

                    self.shrink(selected_clips, 0, (1 / project.fps))

                    if len(selected_clips) == 1:
                        msg = "Shrunk clip from end"
                    else:
                        msg = f"Shrunk {len(selected_clips)} clips from end"

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()

                    project.save_history(msg)
                elif rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                    selected_clips = self.get_selected_clips()

                    self.extend(selected_clips, 0, (1 / project.fps))

                    if len(selected_clips) == 1:
                        msg = "Extended clip from end"
                    else:
                        msg = f"Extended {len(selected_clips)} clips from end"

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()

                    project.save_history(msg)
        elif shift_down:
            if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
                selected_clips = self.get_selected_clips()

                self.shift(selected_clips, (1 / project.fps))

                if len(selected_clips) == 1:
                    msg = "Shifted clip left"
                else:
                    msg = f"Shifted {len(selected_clips)} clips left"

                self.video_buffer = {}
                self.audio_buffer = {}
                self.get_end_time()

                project.save_history(msg)
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                selected_clips = self.get_selected_clips()

                self.shift(selected_clips, -(1 / project.fps))

                if len(selected_clips) == 1:
                    msg = "Shifted clip right"
                else:
                    msg = f"Shifted {len(selected_clips)} clips right"

                self.video_buffer = {}
                self.audio_buffer = {}
                self.get_end_time()

                project.save_history(msg)
        else:
            affected_clip_indexes = []
            outside_timeline = False
            for i in range(len(self.clips)-1, -1, -1):
                if self.clips[i].selected:
                    start_time = self.clips[i].start_time
                    end_time = self.clips[i].end_time

                    if start_time < self.scroll_x or end_time > self.scroll_x + self.width / self.zoom:
                        outside_timeline = True

                    affected_clip_indexes.append(i)

            if rl.is_key_pressed(rl.KeyboardKey.KEY_D) or rl.is_key_pressed(rl.KeyboardKey.KEY_DELETE):
                if outside_timeline and len(affected_clip_indexes) < 100:
                    project.show_message("NOTICE: Did not delete clips outside of timeline view!")
                elif len(affected_clip_indexes) > 0:
                    for i in affected_clip_indexes:
                        if self.clips[i].selected:
                            del self.clips[i]
                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()
                    project.save_history("Delete clip")
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_X):
                if outside_timeline and len(affected_clip_indexes) < 100:
                    project.show_message("NOTICE: Did not extract clips outside of timeline view!")
                elif len(affected_clip_indexes) > 0:
                    for i in affected_clip_indexes:
                        start = self.clips[i].start_time
                        end = self.clips[i].end_time
                        del self.clips[i]
                        # TODO: This operation is slow
                        if len(self.get_clips_between(start, end)) == 0:
                            duration = end - start
                            for clip in self.clips:
                                if clip.start_time > start:
                                    clip.start_time -= duration
                                    clip.end_time -= duration

                    clips_cut = len(affected_clip_indexes)
                    if clips_cut == 1:
                        msg = f"Extracted 1 clip"
                    else:
                        msg = f"Extracted {clips_cut} clips"

                    project.show_message(msg)

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()
                    project.save_history(msg)
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_E):
                if outside_timeline and len(affected_clip_indexes) < 100:
                    project.show_message("NOTICE: Did not modify clips outside of timeline view!")
                elif len(affected_clip_indexes) > 0:
                    selected_clips = self.get_selected_clips()

                    if shift_down:
                        self.shrink(selected_clips, 0.1, 0.1)

                        if len(selected_clips) == 1:
                            msg = "Shrunk clip"
                        else:
                            msg = f"Shrunk {len(selected_clips)} clips"
                    else:
                        self.extend(selected_clips, 0.1, 0.1)

                        if len(selected_clips) == 1:
                            msg = "Extended clip"
                        else:
                            msg = f"Extended {len(selected_clips)} clips"

                    self.video_buffer = {}
                    self.audio_buffer = {}
                    self.get_end_time()

                    project.show_message(msg)
                    project.save_history(msg)
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_G):
                if outside_timeline and len(affected_clip_indexes) < 100:
                    project.show_message("NOTICE: Did not group clips outside of timeline view!")
                elif len(affected_clip_indexes) > 0:
                    if len(affected_clip_indexes) <= 1:
                        project.show_message("ERROR: At least two clips needed for grouping")
                    else:
                        prev_group = None
                        grouped = True
                        for i in affected_clip_indexes:
                            if prev_group is not None and self.clips[i].group != prev_group:
                                grouped = False
                                break
                            prev_group = self.clips[i].group

                        if grouped:
                            for i in affected_clip_indexes:
                                self.clips[i].group = project.group_id
                                project.group_id += 1

                            msg = f"Ungrouped {len(affected_clip_indexes)} clips"
                        else:
                            group = project.group_id
                            project.group_id += 1

                            for i in affected_clip_indexes:
                                self.clips[i].group = group

                            msg = f"Grouped {len(affected_clip_indexes)} clips"

                        project.show_message(msg)
                        project.save_history(msg)

    def extend(self, clips: list[TimelineClip], extend_backward: float, extend_forward: float):
        for clip in clips:
            for overlap_clip in self.clips:
                if overlap_clip != clip and overlap_clip.group != clip.group and overlap_clip.clip == clip.clip:
                    if overlap_clip.start_time >= clip.start_time:
                        clip_end = clip.clip_start + (clip.end_time - clip.start_time)
                        diff = overlap_clip.clip_start - clip_end
                        if diff < 0:
                            extend_forward = 0
                        else:
                            extend_forward = min(extend_forward, diff)
                    if overlap_clip.end_time <= clip.end_time:
                        prev_clip_end = overlap_clip.clip_start + (overlap_clip.end_time - overlap_clip.start_time)
                        diff = clip.clip_start - prev_clip_end
                        if diff < 0:
                            extend_backward = 0
                        else:
                            extend_backward = min(extend_backward, diff)

            if clip.clip_start >= extend_backward:
                clip.clip_start -= extend_backward
            else:
                clip.clip_start = 0

            offset = 0
            max_end = clip.start_time + clip.clip.duration - clip.clip_start
            if clip.end_time + extend_backward + extend_forward < max_end:
                clip.end_time += extend_backward + extend_forward
                offset += extend_backward + extend_forward
            else:
                offset += max_end - clip.end_time
                clip.end_time = max_end

            if offset != 0:
                clips_to_offset = self.get_clips_between(clip.end_time - offset, self.end_time + 1)
                for oclip in clips_to_offset:
                    if oclip != clip and oclip.group != clip.group:
                        oclip.start_time += extend_backward / 2 + extend_forward / 2
                        oclip.end_time += extend_backward / 2 + extend_forward / 2

    def shrink(self, clips: list[TimelineClip], shrink_backward: float, shrink_forward: float):
        for clip in clips:
            offset = 0
            if clip.clip_start <= clip.clip.duration - shrink_backward:
                clip.clip_start += shrink_backward

            min_end = clip.start_time + shrink_forward + shrink_backward
            if clip.end_time > min_end:
                clip.end_time -= shrink_forward + shrink_backward
                offset -= shrink_forward + shrink_backward

            if offset != 0:
                clips_to_offset = self.get_clips_between(clip.end_time - offset, self.end_time + 1)
                for oclip in clips_to_offset:
                    if oclip != clip and oclip.group != clip.group:
                        oclip.start_time += offset / 2
                        oclip.end_time += offset / 2

    def shift(self, clips: list[TimelineClip], amount: float):
        for clip in clips:
            if clip.clip_start + amount >= 0 and (clip.end_time - clip.start_time) <= (clip.clip.duration - (clip.clip_start + amount)):
                clip.clip_start += amount

def timestamp(time_secs: float, fps: int | None = None) -> str:
    hours = int(time_secs / 3600)
    time_secs -= hours * 3600

    minutes = int(time_secs / 60)
    time_secs -= minutes * 60

    seconds = int(time_secs)
    time_secs -= seconds

    if fps:
        fraction = int(time_secs * fps)
    else:
        fraction = int(time_secs * 100)

    return f"{hours:02}:{minutes:02}:{seconds:02}.{fraction:03}"

def create_blank_texture(width: int, height: int) -> rl.Texture:
    image = rl.gen_image_color(width, height, rl.BLANK)
    texture = rl.load_texture_from_image(image)
    rl.unload_image(image)
    return texture

def frame_to_pixels(frame: VideoFrame, width: int, height: int) -> FFI.CData:
    frame_rgba = frame.reformat(width=width, height=height, format="rgba")
    frame_rgba = frame_rgba.planes[0]
    stride = frame_rgba.line_size

    src = memoryview(frame_rgba)
    dest = bytearray(width*4*height)

    for y in range(height):
        dest[y*width*4:(y+1)*width*4] = src[y*stride:y*stride+width*4]

    return rl.ffi.new('char []', bytes(dest))

def bufferer(project: Project):
    while True:
        if project.rendering:
            project.render()
        video_ready = project.build_video_buffer()
        audio_ready = project.build_audio_buffer()
        project.timeline.buffer_ready = video_ready and audio_ready
        time.sleep(0.0001)

def audio_player(project: Project):
    while True:
        # Play audio
        if (project.timeline.is_playing or project.timeline.play_current_audio_frame) and rl.is_audio_stream_processed(project.timeline.audio_stream):
            chunk_number = project.timeline.audio_playhead // project.timeline.audio_framesize

            chunk = bytearray()

            while len(chunk) < project.timeline.audio_bufsize*4:
                if chunk_number in project.timeline.audio_buffer:
                    #print(f"Playing {chunk_number} ({project.audio_playhead})")
                    frame = project.timeline.audio_buffer[chunk_number]
                    if frame is not None:
                        chunk.extend(frame)
                    else:
                        break
                else:
                    break
                if project.timeline.play_current_audio_frame:
                    break
                chunk_number += 1

            chunk.extend(bytearray(project.timeline.audio_bufsize*4 - len(chunk)))

            if not project.timeline.play_current_audio_frame:
                project.timeline.audio_playhead += project.timeline.audio_bufsize

            audio_chunk = rl.ffi.new('char []', bytes(chunk))
            rl.update_audio_stream(project.timeline.audio_stream, audio_chunk, project.timeline.audio_bufsize)
            project.timeline.play_current_audio_frame = False
        time.sleep(0.0000001)

def grapher(project: Project):
    while True:
        for clip in project.clip_bin.clips:
            if clip.clip.audio is not None and clip.clip.audio_graph_progress is None:
                clip.clip.generate_audio_graph()
        time.sleep(0.001)

def main():
    SCALE_FACTOR = 0.8

    project = Project(30, 48000)

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(int(1920*SCALE_FACTOR), int(1080*SCALE_FACTOR), "Video Editor")

    rl.set_target_fps(60)

    video_canvas_width = 1920
    video_canvas_height = 1080

    video_canvas = create_blank_texture(
        video_canvas_width,
        video_canvas_height,
    )

    current_frame_number = 0
    current_audio_frame_number = 0

    grahper_thread = Thread(target=grapher, args=(project,), daemon=True)
    grahper_thread.start()

    bufferer_thread = Thread(target=bufferer, args=(project,), daemon=True)
    bufferer_thread.start()

    audio_player_thread = Thread(target=audio_player, args=(project,), daemon=True)
    audio_player_thread.start()

    render_timeline = True
    render_clip_bin = True
    render_video = True

    loaded_frame = None

    while not rl.window_should_close():
        screen_width = rl.get_screen_width()
        screen_height = rl.get_screen_height()

        mouse_pos = rl.get_mouse_position()
        left_mouse_down = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
        left_mouse_pressed = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)
        left_mouse_released = rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)
        mouse_wheel_move = rl.get_mouse_wheel_move()
        delta_time = rl.get_frame_time()

        ctrl_down = rl.is_key_down(rl.KeyboardKey.KEY_LEFT_CONTROL)
        shift_down = rl.is_key_down(rl.KeyboardKey.KEY_LEFT_SHIFT)

        if rl.is_file_dropped():
            files = rl.load_dropped_files()
            for i in range(files.count):
                dropped_file = str(rl.ffi.string(files.paths[i]))[2:-1]
                project.show_message(f"Dropped in {dropped_file}")

                if dropped_file.endswith(".json"):
                    # TODO: add area on screen where append=True
                    project.load(dropped_file, append=True)
                else:
                    # TODO: detect file type
                    clip = Clip(project, dropped_file, "video")
                    project.clip_bin.add_clip(clip)
            rl.unload_dropped_files(files)
            project.save_history("Dropped files")

        clips = project.timeline.get_current_clips()

        audio = None
        video = None
        for clip in clips:
            if clip.type == "audio":
                audio = clip.resource
            else:
                video = clip.resource

        if video:
            current_frame_number = int(project.timeline.playhead * project.fps)
        current_audio_frame_number = int(project.timeline.playhead * project.audio_fps)

        if ctrl_down:
            if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
                if project.file_path and not shift_down:
                    project.save(project.file_path)
                else:
                    file = open_file_to_save(title="Save Project As...")
                    if file:
                        project.save(file)
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_O):
                file = open_file(title="Open a Project File")
                if file:
                    project.load(file)
                    # TODO: redoing load project screws up timeline
                    #project.save_history("Load new project")
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                file = open_file_to_save(title="Render Project...")
                if file:
                    project.render_file = file
                    project.start_rendering()
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_Z):
                project.undo_history()
            elif rl.is_key_pressed(rl.KeyboardKey.KEY_Y):
                project.redo_history()

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_J):
            project.timeline.jump(-1)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_K):
            project.timeline.jump(1)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_H):
            project.timeline.jump(-60 * 10)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_L):
            project.timeline.jump(60 * 10)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_P):
            project.timeline.jump_to_empty()

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
            project.timeline.jump(-1/project.fps)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
            project.timeline.jump(1/project.fps)

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            render_clip_bin = not render_clip_bin

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_T):
            render_video = not render_video

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_Y):
            render_timeline = not render_timeline

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            project.mode = "select"

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_C):
            project.mode = "cut"

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_M):
            project.mode = "move"

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_KP_ADD):
            project.timeline.audio_offset += 0.01
            project.show_message(f"Audio offset: {project.timeline.audio_offset}")

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_KP_SUBTRACT):
            project.timeline.audio_offset -= 0.01
            project.show_message(f"Audio offset: {project.timeline.audio_offset}")

        elif rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
            project.timeline.buffer_ready = False
            project.timeline.audio_playhead = current_audio_frame_number
            if project.timeline.is_playing or project.timeline.want_to_play:
                project.timeline.is_playing = False
                project.timeline.want_to_play = False
            else:
                project.timeline.want_to_play = True

        cw = video_canvas_width
        ch = video_canvas_height

        video_constrain_width = int(screen_width / 7 * 4)

        if video_constrain_width < cw:
            ratio = ch / cw
            cw = video_constrain_width
            ch = int(cw * ratio)

        if cw != video_canvas.width or ch != video_canvas.height:
            rl.unload_texture(video_canvas)
            video_canvas = create_blank_texture(cw, ch)

        rl.begin_drawing()

        rl.clear_background((10, 10, 20))

        if render_timeline:
            project.timeline.render(0, int(screen_height * (2/3)), screen_width, int(screen_height * (1/3)), mouse_wheel_move, delta_time, ctrl_down, shift_down, left_mouse_pressed, mouse_pos, left_mouse_released, left_mouse_down)

        if render_clip_bin:
            project.clip_bin.render(int(screen_width / 7 * 3)-1, int(screen_height * (2/3)), left_mouse_down, mouse_pos, mouse_wheel_move, delta_time)

        if video:
            if int(current_frame_number) in project.timeline.video_buffer:
                if render_video and int(current_frame_number) != loaded_frame:
                    # Draw video frame
                    video_frame = project.timeline.video_buffer[int(current_frame_number)]
                    if video_frame is not None:
                        frame = frame_to_pixels(video_frame, cw, ch)

                        rl.update_texture(video_canvas, frame)

                        loaded_frame = int(current_frame_number)

            if render_video:
                rl.draw_texture(video_canvas, int(screen_width / 7 * 3), 0, rl.WHITE)

        if project.timeline.is_playing:
            project.timeline.playhead += delta_time
            current_frame_number = project.timeline.playhead * project.fps

            if project.timeline.playhead >= project.timeline.end_time:
                project.timeline.playhead = project.timeline.end_time
                project.timeline.is_playing = False
                project.timeline.want_to_play = False

        rl.draw_fps( 10, screen_height - 50 )

        mode_text = f"MODE: {project.mode.upper()}"
        mode_font_size = 20
        mode_width = rl.measure_text(mode_text, mode_font_size)
        rl.draw_text(mode_text, screen_width - mode_width - 20, screen_height - 50, mode_font_size, rl.ORANGE)

        if project.rendering:
            project.timeline.is_playing = False
            project.timeline.want_to_play = False

            rendering_rect_width = 400
            rendering_rect_height = 100
            rendering_rect = rl.Rectangle(int(screen_width / 2 - rendering_rect_width / 2), int(screen_height / 2 - rendering_rect_height / 2), rendering_rect_width, rendering_rect_height)
            rl.draw_rectangle_rec(rendering_rect, rl.BLACK)
            rl.draw_rectangle_lines_ex(rendering_rect, 5, rl.WHITE)

            rendering_text = "Rendering..."
            rendering_text_size = 40
            rendering_text_width = rl.measure_text(rendering_text, rendering_text_size)
            rl.draw_text(rendering_text, int(screen_width / 2 - rendering_text_width / 2), int(screen_height / 2 - rendering_text_size / 2), rendering_text_size, rl.GREEN)

            pad = 10
            prog_height = 10
            render_progress_rect = rl.Rectangle(
                rendering_rect.x + pad,
                rendering_rect.y + rendering_rect_height - pad - prog_height,
                int((rendering_rect_width - pad * 2) * project.render_progress),
                prog_height,
            )
            rl.draw_rectangle_rec(render_progress_rect, rl.WHITE)

        if project.timeline.want_to_play:
            if project.timeline.buffer_ready:
                project.timeline.want_to_play = False
                project.timeline.is_playing = True

                if project.timeline.is_playing:
                    #print("Playing")
                    rl.play_audio_stream(project.timeline.audio_stream)
                else:
                    #print("Stopped")
                    rl.stop_audio_stream(project.timeline.audio_stream)
            else:
                buffering_rect_width = 300
                buffering_rect_height = 80
                buffering_rect = rl.Rectangle(int(screen_width / 2 - buffering_rect_width / 2), int(screen_height / 2 - buffering_rect_height / 2), buffering_rect_width, buffering_rect_height)
                rl.draw_rectangle_rec(buffering_rect, rl.BLACK)
                rl.draw_rectangle_lines_ex(buffering_rect, 5, rl.WHITE)
                buffering_text = "Buffering..."
                buffering_text_size = 40
                buffering_text_width = rl.measure_text(buffering_text, buffering_text_size)
                rl.draw_text(buffering_text, int(screen_width / 2 - buffering_text_width / 2), int(screen_height / 2 - buffering_text_size / 2), buffering_text_size, rl.VIOLET)

        message_height = 35
        message_margin = 20
        message_font_size = 20
        message_padding = 5
        message_gap = 5
        y = screen_height - message_margin - (message_height + message_gap) * len(project.messages)
        for i in range(len(project.messages)-1, -1, -1):
            message = project.messages[i]
            anim_secs = 0.5
            anim_move = 10
            anim = min(anim_secs, (time.time()-message.time) / anim_secs) * anim_move
            message_width = min(600, rl.measure_text(message.text, message_font_size))
            message_rec = rl.Rectangle(20, y - anim, message_width + message_padding * 2, message_font_size + message_padding * 2)
            rl.draw_rectangle_rec(message_rec, rl.BLACK)
            rl.draw_rectangle_lines_ex(message_rec, 2, rl.ORANGE)
            rl.begin_scissor_mode(int(message_rec.x + message_padding), int(message_rec.y), int(message_rec.width - message_padding * 2), int(message_rec.height))
            rl.draw_text(message.text, int(message_rec.x + message_padding), int(message_rec.y + message_padding), int(message_font_size), rl.ORANGE)
            rl.end_scissor_mode()

            y += message_height + message_gap

            if message.time < time.time()-5:
                del project.messages[i]

        rl.end_drawing()

    rl.unload_texture(video_canvas)
    rl.unload_audio_stream(project.timeline.audio_stream)
    rl.close_window()

if __name__ == "__main__":
    main()
