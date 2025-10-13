#!/usr/bin/python3

import argparse
import functools
import glob
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import typing
from abc import abstractmethod
from collections import UserDict
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


@dataclass(frozen=True)
class Constant:
    VERSION: str = "1.0.7-20251013"
    BASE_FILE_NAME: str = os.path.splitext(os.path.basename(__file__))[0]
    PREF_FILE_NAME: str = BASE_FILE_NAME + ".json"

    SUBCOMMAND_START: str = "start"
    SUBCOMMAND_STOP: str = "stop"
    SUBCOMMAND_RESTART: str = "restart"

    LOG_NAME: str = BASE_FILE_NAME + "_log"
    LOG_WHEN: str = "midnight"
    LOG_INTERVAL: int = 1
    LOG_ENCODING: str = "utf-8"
    LOG_FORMATTER: str = "%(asctime)s:%(levelname)s[%(funcName)s] %(message)s"
    LOG_LEVEL: int = logging.DEBUG

    STREAMLOG_NAME: str = "stream_log"
    STREAMLOG_MAX_BYTES: int = 100 * 1024
    STREAMLOG_ENCODING: str = LOG_ENCODING
    STREAMLOG_FORMATTER: str = "%(asctime)s: %(message)s"
    STREAMLOG_LEVEL: int = logging.WARNING

    _D: str = "[0-9]"
    VFILE_NAME_DATE_FMT: str = "-%Y%m%d-%H%M%S"
    VFILE_NAME_DATE_PAT: str = "-" + (_D * 8) + "-" + (_D * 6)
    ARCHIVE_DIR_NAME: str = "archive"
    ARCHIVE_SUBDIR_NAME_PAT: str = "*-" + (_D * 8)
    ARCHIVE_SUBDIR_NAME_RE: str = r"^(.+-\d{8})-\d{6}\..+$"

    SEC: int = 1
    MIN_TO_SEC: int = 60 * SEC
    HOUR_TO_SEC: int = 60 * MIN_TO_SEC

    MAIN_THREAD_SLEEP: float = float(1 * SEC)
    LOOPER_SLEEP: float = float(1 * SEC)
    STORAGER_EXEC_PERIOD: float = float(5 * MIN_TO_SEC)
    RECORDER_WD_PERIOD: float = float(1 * MIN_TO_SEC)
    RECORDER_OBS_MARGIN: float = float(1 * MIN_TO_SEC)

    LOCK_FILE_NAME: str = BASE_FILE_NAME + ".lock"
    PID_OF_NOBODY: int = -1
    LOCK_CHECK_SLEEP: float = MAIN_THREAD_SLEEP
    LOCK_EXPIRY: float = LOCK_CHECK_SLEEP * 5
    STOP_REQUEST_TIMEOUT: float = float(10 * SEC)

    DATE_FORMAT: str = "%Y/%m/%d %H:%M:%S"
    EMPTY_STR: str = ""


@dataclass(frozen=True)
class Default:
    REC_BIN: str = "ffmpeg"

    LOG_REL_DIR: str = os.path.join("log")
    VIDEO_REL_DIR: str = os.path.join("video")

    LOG_BACKUP: int = 28
    STREAMLOG_BACKUP: int = 5

    ARCHIVING_WAIT_HOUR: int = 6
    REMOVE_TH_MIN: int = 1
    REMOVE_TH_MAX: int = 99
    REMOVE_START: int = REMOVE_TH_MAX
    REMOVE_STOP: int = REMOVE_START

    SEGMENT_SEC: int = 900
    EXT: str = "mp4"


@dataclass(frozen=True)
class Error:
    NONE: int = 0b00000000
    GENERAL: int = 0b00000001
    PREF: int = 0b00000010
    STREAM: int = 0b00000100


C = Constant()
D = Default()
E = Error()
T = typing.TypeVar("T")
LOGGER = logging.getLogger(__name__)
STREAM_LOGGER = logging.getLogger(__name__ + "stream")


class PrefDict(UserDict[typing.Any, typing.Any]):
    def tget(self, key: typing.Any, default: T) -> T:
        try:
            value = super().__getitem__(key)
        except Exception:
            value = default
        return value if isinstance(value, type(default)) else default


@dataclass
class LogEnv:
    cwd: InitVar[str]
    pref: InitVar[PrefDict]

    dir: str = field(init=False)
    log_backup: int = field(init=False)
    streamlog_backup: int = field(init=False)

    def __post_init__(self, cwd: str, pref: PrefDict) -> None:
        self.dir = pref.tget("dir", os.path.join(cwd, D.LOG_REL_DIR))
        self.log_backup = pref.tget("logBackup", D.LOG_BACKUP)
        self.streamlog_backup = pref.tget("streamlogBackup", D.STREAMLOG_BACKUP)


@dataclass
class StoragerEnv:
    cwd: InitVar[str]
    pref: InitVar[PrefDict]

    video_dir: str = field(init=False)
    archive_dir: str = field(init=False)
    archiving_wait: float = field(init=False)
    g_vfile_pat: str = field(init=False)
    remove_start: int = field(init=False)
    remove_stop: int = field(init=False)

    def __post_init__(self, cwd: str, pref: PrefDict) -> None:
        self.video_dir = pref.tget("dir", os.path.join(cwd, D.VIDEO_REL_DIR))
        self.archive_dir = os.path.join(self.video_dir, C.ARCHIVE_DIR_NAME)
        self.archiving_wait = float(
            pref.tget("archivingWaitHour", D.ARCHIVING_WAIT_HOUR) * C.HOUR_TO_SEC
        )
        self.g_vfile_pat = os.path.join(
            self.video_dir, filename("*", C.VFILE_NAME_DATE_PAT, ext="*")
        )

        start = pref.tget("removeStart", D.REMOVE_START)
        stop = pref.tget("removeStop", D.REMOVE_STOP)
        self.remove_start = max(min(start, D.REMOVE_TH_MAX), D.REMOVE_TH_MIN)
        self.remove_stop = min(max(stop, D.REMOVE_TH_MIN), self.remove_start)


@dataclass
class RecorderEnv:
    rec_bin: InitVar[str]
    video_dir: InitVar[str]
    pref: InitVar[PrefDict]

    name: str = field(init=False)
    vfile_pat: str = field(init=False, hash=False, compare=False)
    obs_time: float = field(init=False, hash=False, compare=False)
    cmd: tuple[str, ...] = field(init=False, hash=False, compare=False)

    def __post_init__(self, rec_bin: str, video_dir: str, pref: PrefDict) -> None:
        name = pref.get("name")
        url = pref.get("url")
        if isinstance(name, str) and isinstance(url, str):
            ext = pref.tget("ext", D.EXT)
            vcodec = pref.get("vcodec")
            fps = pref.get("fps")
            acodec = pref.get("acodec")
            segment_sec = pref.tget("segmentSec", D.SEGMENT_SEC)

            vcodec_option = ["-c:v", vcodec] if isinstance(vcodec, str) else []
            fps_option = ["-r", str(fps)] if isinstance(fps, int) else []
            acodec_option = ["-c:a", acodec] if isinstance(acodec, str) else []
            vfile = os.path.join(
                video_dir, filename(name, C.VFILE_NAME_DATE_FMT, ext=ext)
            )

            self.name = name
            self.vfile_pat = os.path.join(
                video_dir, filename(name, C.VFILE_NAME_DATE_PAT, ext=ext)
            )
            self.obs_time = float(segment_sec) + C.RECORDER_OBS_MARGIN
            self.cmd = tuple(
                [rec_bin]
                + ["-nostdin"]
                + ["-hide_banner"]
                + ["-loglevel", "warning"]
                + ["-i", url]
                + vcodec_option
                + fps_option
                + acodec_option
                + ["-f", "segment"]
                + ["-segment_time", str(segment_sec)]
                + ["-reset_timestamps", "1"]
                + ["-segment_atclocktime", "1"]
                + ["-strftime", "1"]
                + [vfile]
            )
        else:
            raise TypeError(f"INVALID PARAMETER: {pref}")


@dataclass
class Env:
    cwd: InitVar[str]

    lock: str = field(init=False)
    log_env: LogEnv = field(init=False)
    storager_env: StoragerEnv = field(init=False)
    recorder_envs: list[RecorderEnv] = field(init=False)

    def __post_init__(self, cwd: str) -> None:
        with open(os.path.join(cwd, C.PREF_FILE_NAME)) as f:
            pref = PrefDict(json.load(f))

        self.lock = os.path.join(cwd, C.LOCK_FILE_NAME)
        self.log_env = LogEnv(cwd, PrefDict(pref.get("log")))
        self.storager_env = StoragerEnv(cwd, PrefDict(pref.get("video")))

        rec_bin = pref.tget("recBin", D.REC_BIN)
        seen: list[str] = []
        self.recorder_envs = []
        for stream in pref.get("streams", []):
            if isinstance(name := stream.get("name"), str) and name not in seen:
                seen.append(name)
                self.recorder_envs.append(
                    RecorderEnv(rec_bin, self.storager_env.video_dir, PrefDict(stream))
                )


class Looper(threading.Thread):
    def __init__(self, caller: threading.Thread) -> None:
        self._caller: threading.Thread = caller
        self._is_stopping: bool = False
        self._basetime: float = 0.0
        super().__init__(daemon=False)

    def run(self) -> None:
        self.reset_elapsed()
        self.__initial__()
        while self._caller.is_alive() and (not self._is_stopping):
            self.__loop__()
            time.sleep(C.LOOPER_SLEEP)
        self.__final__()

    @abstractmethod
    def __initial__(self) -> None:
        pass

    @abstractmethod
    def __loop__(self) -> None:
        pass

    @abstractmethod
    def __final__(self) -> None:
        pass

    @property
    def elapsed(self) -> float:
        return time.time() - self._basetime

    def reset_elapsed(self) -> None:
        self._basetime = time.time()

    def send_stop(self) -> None:
        self._is_stopping = True


class Recorder(Looper):
    def __init__(self, caller: threading.Thread, env: RecorderEnv) -> None:
        self._env: RecorderEnv = env
        self._popen: subprocess.Popen[str] | None = None
        self._stream_logger_thread: threading.Thread | None = None
        super().__init__(caller)

    def __initial__(self) -> None:
        self._popen = subprocess.Popen(
            self._env.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        LOGGER.info(f"start Recorder: {self._env.name}")

        self._stream_logger_thread = threading.Thread(
            target=self._stream_logger, args=(self._popen, self._env.name), daemon=True
        )
        self._stream_logger_thread.start()

    def __loop__(self) -> None:
        if self._popen and (self._popen.poll() is None):
            if self.elapsed >= C.RECORDER_WD_PERIOD:
                if not is_recording(self._env.vfile_pat, self._env.obs_time):
                    LOGGER.error(f"KILL INACTIVE STREAM: {self._env.name}")
                    self.send_stop()
                self.reset_elapsed()
        else:
            self.send_stop()

    def __final__(self) -> None:
        returncode: int | None = None
        if self._popen:
            self._popen.kill()
            returncode = self._popen.wait()
        LOGGER.info(f"stop Recorder: {self._env.name}({returncode})")

        if self._stream_logger_thread:
            self._stream_logger_thread.join()

    def _stream_logger(self, popen: subprocess.Popen[str], stream_name: str) -> None:
        LOGGER.info("start stream logger")
        while popen.stderr and (popen.poll() is None):
            STREAM_LOGGER.warning(f"[{stream_name}] {popen.stderr.readline().rstrip()}")
        LOGGER.info("stop stream logger")

    @property
    def env(self) -> RecorderEnv:
        return self._env


class Storager(Looper):
    def __init__(self, caller: threading.Thread, env: StoragerEnv) -> None:
        self._env: StoragerEnv = env
        super().__init__(caller)

    def __initial__(self) -> None:
        LOGGER.info("start Storager")

    def __loop__(self) -> None:
        if self.elapsed >= C.STORAGER_EXEC_PERIOD:
            self._rm_subdir(
                self._env.archive_dir,
                C.ARCHIVE_SUBDIR_NAME_PAT,
                self._env.remove_start,
                self._env.remove_stop,
            )
            self._to_subdir(
                self._env.g_vfile_pat,
                self._env.archiving_wait,
                self._env.archive_dir,
                C.ARCHIVE_SUBDIR_NAME_RE,
            )
            self.reset_elapsed()

    def __final__(self) -> None:
        LOGGER.info("stop Storager")

    def _dst(self, file: str, dir: str, subdir_name_re: str) -> str | None:
        file_name = os.path.basename(file)
        match = re.search(subdir_name_re, file_name)
        return os.path.join(dir, match.group(1), file_name) if match else None

    def _to_subdir(self, pat: str, mtime: float, dir: str, subdir_name_re: str) -> None:
        for src, dst in [
            (file, self._dst(file, dir, subdir_name_re)) for file in find(pat, mtime)
        ]:
            LOGGER.info(f"archiving: {src} -> {dst}")
            if not move_file(src, dst):
                LOGGER.error(f"FAILED TO ARCHIVE: {src} -> {dst}")

    def _rm_subdir(self, dir: str, subdir_name_pat: str, start: int, stop: int) -> None:
        total, used = usage(dir)
        if (current := usage_rate(total, used)) >= start:
            LOGGER.info(f"start: {current} >= {start}")
            for subdir in find(os.path.join(dir, subdir_name_pat), type="d", sort="t"):
                LOGGER.info(f"removing: {subdir}")
                if isinstance(rm_size := rmdir(subdir), int):
                    used -= rm_size
                else:
                    LOGGER.error(f"FAILED TO REMOVE: {subdir}")
                if (current := usage_rate(total, used)) < stop:
                    LOGGER.info(f"stop: {current} < {stop}")
                    break
            total, used = usage(dir)
            if (current := usage_rate(total, used)) >= stop:
                LOGGER.warning(f"Not Reached: {current} >= {stop}")


LooperT = typing.TypeVar("LooperT", bound=Looper)


def filename(*parts: str, ext: str) -> str:
    filename = C.EMPTY_STR

    for part in parts:
        filename += part
    filename += "." + ext

    return filename


def read_file(file: str) -> str:
    chars: str = C.EMPTY_STR

    try:
        with open(file) as f:
            chars = f.read()
    except Exception as e:
        LOGGER.error(f"FAILED TO READ: {file} - {e}")

    return chars


def write_file(file: str, s: str) -> int:
    n_chars: int = -1

    try:
        with open(file, "w") as f:
            n_chars = f.write(s)
    except Exception as e:
        LOGGER.error(f"FAILED TO WRITE: {file} - {e}")

    return n_chars


def rm(path: str) -> int | None:
    size: int | None = None

    if os.path.isfile(path):
        size = os.path.getsize(path)
        try:
            os.remove(path)
        except Exception as e:
            size = None
            LOGGER.error(f"FAILED TO REMOVE FILE: {path} - {e}")
    else:
        LOGGER.error(f"FILE NOT FOUND: {path}")

    return size


def rmdir(path: str) -> int | None:
    size: int | None = None

    if os.path.isdir(path):
        temp_size: int | None = None
        size = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    temp_size = rm(entry.path)
                elif entry.is_dir():
                    temp_size = rmdir(entry.path)
                else:
                    LOGGER.error(f"UNEXPECTED PATH: {path}")

                if isinstance(temp_size, int):
                    size += temp_size

            os.rmdir(path)
        except Exception as e:
            LOGGER.error(f"FAILED TO REMOVE DIRECTORY: {path} - {e}")
    else:
        LOGGER.error(f"DIRECTORY NOT FOUND: {path}")

    return size


def makedirs(dir: str) -> bool:
    result: bool = False

    if not os.path.isdir(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            result = True
        except Exception as e:
            LOGGER.error(f"FAILED TO MAKE DIRECTORY: {dir} - {e}")
    else:
        result = True

    return result


def move_file(src: str | None, dst: str | None) -> str | None:
    path: str | None = None

    if isinstance(src, str) and isinstance(dst, str):
        if (not os.path.isdir(dst)) and makedirs(os.path.dirname(dst)):
            try:
                path = shutil.move(src, dst)
            except Exception as e:
                LOGGER.error(f"FAILED TO MOVE: {src} -> {dst} - {e}")

    return path


FindType = typing.Literal["f", "d", "fd"]
FindSort = typing.Literal["n", "t"]


def find(
    pat: str, mtime: float | None = None, type: FindType = "f", sort: FindSort = "n"
) -> list[str]:
    def _mtime_sorted(_paths: list[str]) -> list[str]:
        return sorted(_paths, key=os.path.getmtime)

    def _is_mtime_match(_basetime: float, _mtime: float, _path: str) -> bool:
        return ((diff := _basetime - os.path.getmtime(_path)) >= 0) and (
            ((_mtime >= 0) and (diff >= _mtime)) or ((_mtime < 0) and (diff <= -_mtime))
        )

    do_sort = sorted
    if sort == "t":
        do_sort = _mtime_sorted

    is_type_match = bool
    if type == "f":
        is_type_match = os.path.isfile
    elif type == "d":
        is_type_match = os.path.isdir

    is_mtime_match = bool
    if mtime:
        is_mtime_match = functools.partial(_is_mtime_match, time.time(), mtime)

    def is_match(path: str) -> bool:
        return is_type_match(path) and is_mtime_match(path)

    return do_sort([path for path in glob.glob(pat) if is_match(path)])


def usage(dir: str) -> tuple[int, int]:
    total: int = 0
    used: int = 0

    try:
        total, used, _ = shutil.disk_usage(dir)
    except Exception as e:
        LOGGER.error(f"FAILED TO GET USAGE: {dir} - {e}")

    return (total, used)


def usage_rate(total: int, used: int) -> int:
    return int((used * 100) / total) if ((total > 0) and (used >= 0)) else -1


def acquire_lock(lock: str, pid: int) -> bool:
    result: bool = False

    is_acquirable: bool = True
    if os.path.isfile(lock) and (write_file(lock, C.EMPTY_STR) == len(C.EMPTY_STR)):
        t = time.time()
        while (time.time() - t) < C.LOCK_EXPIRY:
            if os.path.getsize(lock) > 0:
                is_acquirable = False
                break
            time.sleep(C.LOCK_CHECK_SLEEP)

    if is_acquirable:
        result = write_file(lock, str(pid)) > 0

    return result


def update_lock(lock: str, pid: int) -> bool:
    result: bool = False

    if (pid_str := read_file(lock)) == C.EMPTY_STR:
        result = write_file(lock, str(pid)) > 0
    else:
        try:
            result = int(pid_str) == pid
        except Exception as e:
            LOGGER.error(f"UNEXPECTED CONTENT IN LOCK: {pid_str} - {e}")

    return result


def release_lock(lock: str) -> int | None:
    return rm(lock)


def stop_request(lock: str) -> bool | None:
    result: bool | None = None

    if os.path.isfile(lock):
        result = False
        if write_file(lock, str(C.PID_OF_NOBODY)) > 0:
            t = time.time()
            while (time.time() - t) < C.STOP_REQUEST_TIMEOUT:
                if not os.path.isfile(lock):
                    result = True
                    break
                time.sleep(C.LOCK_CHECK_SLEEP)
            if not result and (read_file(lock) == str(C.PID_OF_NOBODY)):
                result = isinstance(release_lock(lock), int)
        else:
            LOGGER.error(f"FAILED STOP REQUEST: {lock}")

    return result


def is_recording(vfile_pat: str, observation_time: float) -> bool:
    return bool(find(vfile_pat, -observation_time))


def start_looper(looper: LooperT | None) -> LooperT | None:
    return looper if looper and (not looper.start()) else None


def stop_looper(looper: Looper | None) -> None:
    if looper:
        looper.send_stop()
        try:
            looper.join()
        except Exception:
            pass


def start_loopers(loopers: list[LooperT]) -> list[LooperT]:
    return [looper for looper in loopers if not looper.start()]


def stop_loopers(loopers: list[LooperT]) -> None:
    for stopping_looper in [looper for looper in loopers if not looper.send_stop()]:
        try:
            stopping_looper.join()
        except Exception:
            pass


def is_alive(looper: Looper | None) -> bool:
    return looper.is_alive() if looper else False


def get_alive_loopers(loopers: list[LooperT]) -> list[LooperT]:
    return [looper for looper in loopers if looper.is_alive()]


def get_unwanted(
    recorders: list[Recorder], recorder_envs: list[RecorderEnv]
) -> list[Recorder]:
    unwanted_dict: dict[int, Recorder] = {}

    for i in range(len(recorders)):
        if i not in unwanted_dict:
            if recorders[i].env not in recorder_envs:
                unwanted_dict[i] = recorders[i]
            else:
                duplicated_dict: dict[int, Recorder] = {
                    i + j: recorders[i + j]
                    for j in range(len(recorders[i:]))
                    if recorders[i + j].env == recorders[i].env
                }
                if len(duplicated_dict) > 1:
                    unwanted_dict |= duplicated_dict

    return [record for record in unwanted_dict.values()]


def get_unrecordings(
    recorder_envs: list[RecorderEnv], recorders: list[Recorder]
) -> list[RecorderEnv]:
    recording_envs = [recorder.env for recorder in recorders if recorder]
    return [
        recorder_env
        for recorder_env in recorder_envs
        if recorder_env not in recording_envs
    ]


def status(env: Env, args: argparse.Namespace) -> int:
    result: int = E.NONE

    print(datetime.now().strftime(C.DATE_FORMAT))
    print(C.EMPTY_STR)

    print("[STREAM]")
    for recorder_env in env.recorder_envs:
        if is_recording(recorder_env.vfile_pat, recorder_env.obs_time):
            print(f"{recorder_env.name} : OK")
        else:
            print(f"{recorder_env.name} : NG")
            result = result | E.STREAM
    print(C.EMPTY_STR)

    print("[STORAGE]")
    total, used = usage(env.storager_env.video_dir)
    if (current := usage_rate(total, used)) >= 0:
        print(f"Current : {current}%")
        print(f"Remove start : {env.storager_env.remove_start}%")
        print(f"Remove stop : {env.storager_env.remove_stop}%")
    else:
        LOGGER.error("FAILED TO GET DISK USAGE")
        result = result | E.GENERAL
    print(C.EMPTY_STR)

    return result


def start(env: Env, args: argparse.Namespace) -> int:
    result: int = E.NONE

    LOGGER.info("in")

    main_thread = threading.main_thread()
    storager: Storager | None = None
    recorders: list[Recorder] = []
    pid: int = os.getpid()

    if acquire_lock(env.lock, pid):
        while update_lock(env.lock, pid):
            if not is_alive(storager):
                storager = start_looper(Storager(main_thread, env.storager_env))

            stop_loopers(get_unwanted(recorders, env.recorder_envs))
            recorders = get_alive_loopers(recorders)
            recorders += start_loopers(
                [
                    Recorder(main_thread, recorder_env)
                    for recorder_env in get_unrecordings(env.recorder_envs, recorders)
                ]
            )
            if not recorders:
                LOGGER.warning("No Stream To Record")
                result = E.STREAM
                break

            time.sleep(C.MAIN_THREAD_SLEEP)

        stop_loopers(recorders)
        stop_looper(storager)
        if not release_lock(env.lock):
            LOGGER.error("FAILED TO RELEASE LOCK")

    else:
        LOGGER.warning("Already Running")
        result = E.GENERAL

    LOGGER.info(f"out({result})")

    return result


def stop(env: Env, args: argparse.Namespace) -> int:
    result: int = E.NONE

    LOGGER.info("in")

    if isinstance(response := stop_request(env.lock), bool):
        if not response:
            LOGGER.error("FAILED TO STOP")
            result = E.GENERAL
    else:
        LOGGER.warning("Not Running")
        result = E.GENERAL

    LOGGER.info(f"out({result})")

    return result


def restart(env: Env, args: argparse.Namespace) -> int:
    result: int = E.NONE

    if (result := stop(env, args)) == E.NONE:
        result = start(env, args)
    else:
        LOGGER.error("FAILED TO RESTART")

    return result


def init(cwd: str) -> Env | None:
    env: Env | None = None

    is_made_dir: bool = True
    try:
        env = Env(cwd)
        for dir in [env.log_env.dir, env.storager_env.archive_dir]:
            is_made_dir &= makedirs(dir)
    except Exception as e:
        is_made_dir = False
        print(f"FAILED TO INITIALIZE: {cwd} - {e}", file=sys.stderr)

    if env and is_made_dir:
        handler = TimedRotatingFileHandler(
            os.path.join(env.log_env.dir, C.LOG_NAME),
            when=C.LOG_WHEN,
            backupCount=env.log_env.log_backup,
            interval=C.LOG_INTERVAL,
            encoding=C.LOG_ENCODING,
        )
        handler.setFormatter(logging.Formatter(C.LOG_FORMATTER))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(C.LOG_LEVEL)

        stream_handler = RotatingFileHandler(
            os.path.join(env.log_env.dir, C.STREAMLOG_NAME),
            maxBytes=C.STREAMLOG_MAX_BYTES,
            backupCount=env.log_env.streamlog_backup,
            encoding=C.STREAMLOG_ENCODING,
        )
        stream_handler.setFormatter(logging.Formatter(C.STREAMLOG_FORMATTER))
        STREAM_LOGGER.addHandler(stream_handler)
        STREAM_LOGGER.setLevel(C.STREAMLOG_LEVEL)

    else:
        env = None
        print("FAILED TO PREPARE DIRECTORIES", file=sys.stderr)

    return env


def main() -> int:
    result: int = E.NONE

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=status)
    parser.add_argument("-v", "--version", action="version", version=C.VERSION)
    parser.add_argument("-d", "--dir", default=os.getcwd())
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser(C.SUBCOMMAND_START)
    subparser.set_defaults(func=start)

    subparser = subparsers.add_parser(C.SUBCOMMAND_STOP)
    subparser.set_defaults(func=stop)

    subparser = subparsers.add_parser(C.SUBCOMMAND_RESTART)
    subparser.set_defaults(func=restart)
    args = parser.parse_args()

    if isinstance(env := init(str(pathlib.Path(args.dir).resolve())), Env):
        try:
            result = args.func(env, args)
        except Exception as e:
            LOGGER.critical(f"UNEXPECTED ERROR: {e}")
            result = E.GENERAL
    else:
        result = E.PREF

    return result


if __name__ == "__main__":
    sys.exit(main())
