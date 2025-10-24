#!/usr/bin/python3

import argparse
import functools
import json
import logging
import os
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
from enum import Enum, auto
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path


@dataclass(frozen=True)
class Constant:
    VERSION: str = "1.0.10-20251019-mod"
    BASE_FILE_NAME: str = Path(__file__).stem
    PREF_FILE_NAME: str = BASE_FILE_NAME + ".json"
    LOCK_FILE_NAME: str = BASE_FILE_NAME + ".lock"

    D: str = "[0-9]"
    KB: int = 1024
    SEC: int = 1
    MIN: int = 60 * SEC
    HOUR: int = 60 * MIN

    SUBCOMMAND_START: str = "start"
    SUBCOMMAND_STOP: str = "stop"

    LOCK_CMD_TEXT_CHECK: str = "CHECK"
    LOCK_CMD_TEXT_STOP: str = "STOP"

    LOG_NAME: str = BASE_FILE_NAME + "_log"
    LOG_WHEN: str = "midnight"
    LOG_INTERVAL: int = 1
    LOG_ENCODING: str = "utf-8"
    LOG_FORMATTER: str = "%(asctime)s:%(levelname)s[%(funcName)s] %(message)s"
    LOG_LEVEL: int = logging.DEBUG

    STREAMLOG_NAME: str = "stream_log"
    STREAMLOG_ENCODING: str = LOG_ENCODING
    STREAMLOG_FORMATTER: str = "%(asctime)s: %(message)s"
    STREAMLOG_LEVEL: int = logging.WARNING

    VFILE_NAME_DATE_FMT: str = "-%Y%m%d-%H%M%S"
    VFILE_NAME_DATE_PAT: str = "-" + (D * 8) + "-" + (D * 6)
    ARCHIVE_DIR_NAME: str = "archive"
    ARCHIVE_SUBDIR_NAME_PAT: str = "*-" + (D * 8)
    ARCHIVE_SUBDIR_NAME_RE: str = r"^(.+-\d{8})-\d{6}\..+$"

    MAIN_THREAD_SLEEP: float = float(1 * SEC)
    LOOPER_SLEEP: float = float(1 * SEC)
    STORAGER_EXEC_PERIOD: float = float(5 * MIN)
    RECORDER_WD_PERIOD: float = float(1 * MIN)
    RECORDER_OBS_MARGIN: float = float(1 * MIN)
    RECBIN_TERM_TIMEOUT: float = float(5 * SEC)

    DATE_FORMAT: str = "%Y/%m/%d %H:%M:%S"
    EMPTY_STR: str = ""


@dataclass(frozen=True)
class Default:
    REC_BIN: str = "ffmpeg"

    LOG_REL_DIR: Path = Path("log")
    VIDEO_REL_DIR: Path = Path("video")

    LOG_BACKUP: int = 28
    STREAMLOG_SIZE_KB: int = 100
    STREAMLOG_BACKUP: int = 5

    ARCHIVING_WAIT_HOUR: int = 6
    REMOVE_TH_MIN: int = 1
    REMOVE_TH_MAX: int = 99
    REMOVE_START: int = REMOVE_TH_MAX
    REMOVE_STOP: int = REMOVE_START

    TRANSPORT: str = "udp"
    EXT: str = "mp4"
    SEGMENT_SEC: int = 900


@dataclass(frozen=True)
class ErrorCode:
    NONE: int = 0b00000000
    GENERAL: int = 0b00000001
    PREF: int = 0b00000010
    STREAM: int = 0b00000100


C = Constant()
D = Default()
E = ErrorCode()
LOGGER = logging.getLogger(__name__)
STREAM_LOGGER = logging.getLogger(__name__ + "stream")
T = typing.TypeVar("T")


class PrefDict(UserDict[typing.Any, typing.Any]):
    def tget(self, key: typing.Any, default: T) -> T:
        value = self.get(key)
        return value if isinstance(value, type(default)) else default


@dataclass
class LogEnv:
    cwd: InitVar[Path]
    pref: InitVar[PrefDict]

    log_dir: Path = field(init=False)
    log_backup: int = field(init=False)
    streamlog_size: int = field(init=False)
    streamlog_backup: int = field(init=False)

    def __post_init__(self, cwd: Path, pref: PrefDict) -> None:
        self.log_dir = Path(pref.tget("dir", str(cwd / D.LOG_REL_DIR)))
        self.log_backup = pref.tget("logBackup", D.LOG_BACKUP)
        self.streamlog_size = pref.tget("streamlogSizeKb", D.STREAMLOG_SIZE_KB) * C.KB
        self.streamlog_backup = pref.tget("streamlogBackup", D.STREAMLOG_BACKUP)


@dataclass
class StoragerEnv:
    cwd: InitVar[Path]
    pref: InitVar[PrefDict]

    video_dir: Path = field(init=False)
    archive_dir: Path = field(init=False)
    archiving_wait: float = field(init=False)
    g_vfile_name_pat: str = field(init=False)
    remove_start: int = field(init=False)
    remove_stop: int = field(init=False)

    def __post_init__(self, cwd: Path, pref: PrefDict) -> None:
        self.video_dir = Path(pref.tget("dir", str(cwd / D.VIDEO_REL_DIR)))
        self.archive_dir = self.video_dir / C.ARCHIVE_DIR_NAME
        self.archiving_wait = float(
            pref.tget("archivingWaitHour", D.ARCHIVING_WAIT_HOUR) * C.HOUR
        )
        self.g_vfile_name_pat = filename("*", C.VFILE_NAME_DATE_PAT, ext="*")

        start: int = pref.tget("removeStart", D.REMOVE_START)
        stop: int = pref.tget("removeStop", D.REMOVE_STOP)
        self.remove_start = max(min(start, D.REMOVE_TH_MAX), D.REMOVE_TH_MIN)
        self.remove_stop = min(max(stop, D.REMOVE_TH_MIN), self.remove_start)


@dataclass
class RecorderEnv:
    rec_bin: InitVar[str]
    dir: InitVar[Path]
    pref: InitVar[PrefDict]

    name: str = field(init=False)
    video_dir: Path = field(init=False, hash=False, compare=False)
    vfile_name_pat: str = field(init=False, hash=False, compare=False)
    obs_time: float = field(init=False, hash=False, compare=False)
    cmd: tuple[str, ...] = field(init=False, hash=False, compare=False)

    def __post_init__(self, rec_bin: str, dir: Path, pref: PrefDict) -> None:
        name: typing.Any = pref.get("name")
        url: typing.Any = pref.get("url")
        if isinstance(name, str) and isinstance(url, str):
            transport: str = pref.tget("transport", D.TRANSPORT)
            ext: str = pref.tget("ext", D.EXT)
            vcodec: typing.Any = pref.get("vcodec")
            fps: typing.Any = pref.get("fps")
            acodec: typing.Any = pref.get("acodec")
            segment_sec: int = pref.tget("segmentSec", D.SEGMENT_SEC)

            self.name = name
            self.video_dir = dir
            self.vfile_name_pat = filename(name, C.VFILE_NAME_DATE_PAT, ext=ext)
            self.obs_time = float(segment_sec) + C.RECORDER_OBS_MARGIN
            self.cmd = tuple(
                [rec_bin]
                + ["-nostdin"]
                + ["-hide_banner"]
                + ["-loglevel", "warning"]
                + ["-rtsp_transport", transport]
                + ["-i", url]
                + (["-c:v", vcodec] if isinstance(vcodec, str) else [])
                + (["-r", str(fps)] if isinstance(fps, int) else [])
                + (["-c:a", acodec] if isinstance(acodec, str) else [])
                + ["-f", "segment"]
                + ["-segment_time", str(segment_sec)]
                + ["-reset_timestamps", "1"]
                + ["-segment_atclocktime", "1"]
                + ["-strftime", "1"]
                + [str(dir / filename(name, C.VFILE_NAME_DATE_FMT, ext=ext))]
            )
        else:
            raise TypeError(f"INVALID PARAMETER: {pref}")


@dataclass
class Env:
    cwd: InitVar[Path]

    log_env: LogEnv = field(init=False)
    storager_env: StoragerEnv = field(init=False)
    recorder_envs: list[RecorderEnv] = field(init=False)

    def __post_init__(self, cwd: Path) -> None:
        if isinstance(json_str := read_text(cwd / C.PREF_FILE_NAME), str):
            pref: PrefDict = PrefDict(json.loads(json_str))

            self.log_env = LogEnv(cwd, PrefDict(pref.get("log")))
            self.storager_env = StoragerEnv(cwd, PrefDict(pref.get("video")))

            rec_bin: str = pref.tget("recBin", D.REC_BIN)
            recorder_env_dict: dict[str, RecorderEnv] = {}
            for stream in pref.get("streams", []):
                stream_pref: PrefDict = PrefDict(stream)
                if isinstance(name := stream_pref.get("name"), str):
                    recorder_env_dict[name] = RecorderEnv(
                        rec_bin, self.storager_env.video_dir, stream_pref
                    )
            self.recorder_envs = list(recorder_env_dict.values())
        else:
            raise ValueError(f"INVALID JSON: {cwd / C.PREF_FILE_NAME}")


class Lock:
    @dataclass(frozen=True)
    class _Cmd:
        text: str
        release: bool

    class _Res(Enum):
        NOT_AVAILABLE = auto()
        FAILED = auto()
        NO_RESPONSE = auto()
        ACCEPTED = auto()

    _CMD_CHECK: _Cmd = _Cmd(C.LOCK_CMD_TEXT_CHECK, False)
    _CMD_STOP: _Cmd = _Cmd(C.LOCK_CMD_TEXT_STOP, True)
    _SLEEP: float = C.MAIN_THREAD_SLEEP
    _TIMEOUT: float = _SLEEP * 5

    def __init__(self, cwd: Path) -> None:
        self._lock_file = cwd / C.LOCK_FILE_NAME

    def _issue(self, cmd: _Cmd) -> _Res:
        res: Lock._Res = Lock._Res.NOT_AVAILABLE

        if self._lock_file.is_file():
            if write_text(self._lock_file, cmd.text):
                res = Lock._Res.NO_RESPONSE
                t: float = time.time()
                while (time.time() - t) < Lock._TIMEOUT:
                    if (cmd.release and (not self._lock_file.exists())) or (
                        not cmd.release and (read_text(self._lock_file) != cmd.text)
                    ):
                        res = Lock._Res.ACCEPTED
                        break
            else:
                LOGGER.error("FAILED TO OPERATE LOCK FILE")
                res = Lock._Res.FAILED

        return res

    def check(self) -> bool | None:
        result: bool | None = None

        if (res := self._issue(Lock._CMD_CHECK)) == Lock._Res.ACCEPTED:
            result = True
        elif (res == Lock._Res.NOT_AVAILABLE) or (res == Lock._Res.NO_RESPONSE):
            result = False

        return result

    def stop(self) -> bool | None:
        result: bool | None = None

        if (res := self._issue(Lock._CMD_STOP)) == Lock._Res.ACCEPTED:
            result = True
        elif res == Lock._Res.FAILED:
            result = False
        elif res == Lock._Res.NO_RESPONSE:
            result = self.release()

        return result

    def acquire(self, pid: int) -> bool:
        is_acquired: bool = False

        if isinstance((result := self.check()), bool) and (not result):
            is_acquired = write_text(self._lock_file, str(pid))

        return is_acquired

    def release(self) -> bool:
        return remove(self._lock_file)

    def update(self, pid: int) -> bool:
        is_updated: bool = False

        if (lock_text := read_text(self._lock_file)) == str(pid):
            is_updated = True
        elif lock_text == Lock._CMD_CHECK.text:
            is_updated = write_text(self._lock_file, str(pid))
        elif lock_text == Lock._CMD_STOP.text:
            LOGGER.info("issued stop command")
        else:
            LOGGER.error(f"UNEXPECTED TEXT IN LOCK FILE: {lock_text}")

        return is_updated


class Looper(threading.Thread):
    def __init__(self, caller: threading.Thread) -> None:
        self._caller: threading.Thread = caller
        self._is_stopping: bool = False
        self._basetime: float = 0.0
        super().__init__(daemon=False)

    def run(self) -> None:
        self.reset_elapsed()
        self._setup()
        while self._caller.is_alive() and (not self._is_stopping):
            self._loop()
            time.sleep(C.LOOPER_SLEEP)
        self._teardown()

    @abstractmethod
    def _setup(self) -> None:
        pass

    @abstractmethod
    def _loop(self) -> None:
        pass

    @abstractmethod
    def _teardown(self) -> None:
        pass

    @property
    def elapsed(self) -> float:
        return time.time() - self._basetime

    def reset_elapsed(self) -> None:
        self._basetime = time.time()

    def send_stop(self) -> None:
        self._is_stopping = True


class Storager(Looper):
    def __init__(self, caller: threading.Thread, env: StoragerEnv) -> None:
        self._env: StoragerEnv = env
        super().__init__(caller)

    def _setup(self) -> None:
        LOGGER.info("start Storager")

    def _loop(self) -> None:
        if self.elapsed >= C.STORAGER_EXEC_PERIOD:
            self._rm_subdir(
                self._env.archive_dir,
                C.ARCHIVE_SUBDIR_NAME_PAT,
                self._env.remove_start,
                self._env.remove_stop,
            )
            self._to_subdir(
                self._env.g_vfile_name_pat,
                self._env.video_dir,
                self._env.archiving_wait,
                self._env.archive_dir,
                C.ARCHIVE_SUBDIR_NAME_RE,
            )
            self.reset_elapsed()

    def _teardown(self) -> None:
        LOGGER.info("stop Storager")

    def _dst(self, file_name: str, dir: Path, subdir_name_re: str) -> Path | None:
        match: re.Match[str] | None = re.search(subdir_name_re, file_name)
        return (dir / match.group(1) / file_name) if (match is not None) else None

    def _to_subdir(
        self, pat: str, src_dir: Path, mtime: float, dst_dir: Path, subdir_name_re: str
    ) -> None:
        for src, dst in [
            (file, self._dst(file.name, dst_dir, subdir_name_re))
            for file in find(pat, src_dir, mtime)
        ]:
            LOGGER.info(f"archiving: {src} -> {dst}")
            if not move_file(src, dst):
                LOGGER.error(f"FAILED TO ARCHIVE: {src} -> {dst}")

    def _rm_subdir(
        self, dir: Path, subdir_name_pat: str, start: int, stop: int
    ) -> None:
        if isinstance((current := usage_rate(dir)), int) and (current >= start):
            LOGGER.info(f"start: {current} >= {start}")

            for subdir in find(subdir_name_pat, dir, type="d", sort="t"):
                LOGGER.info(f"removing: {subdir}")

                if not remove(subdir):
                    LOGGER.error(f"FAILED TO REMOVE: {subdir}")

                if isinstance((current := usage_rate(dir)), int) and (current < stop):
                    LOGGER.info(f"stop: {current} < {stop}")
                    break
                elif not isinstance(current, int):
                    LOGGER.error("UNCONTINUABLE - DISK USAGE UNKNOWN")
                    break

            if isinstance(current, int) and (current >= stop):
                LOGGER.warning(f"Not Reached: {current} >= {stop}")

        elif not isinstance(current, int):
            LOGGER.error("UNEXECUTABLE - DISK USAGE UNKNOWN")


class Recorder(Looper):
    def __init__(self, caller: threading.Thread, env: RecorderEnv) -> None:
        self._env: RecorderEnv = env
        self._popen: subprocess.Popen[str] | None = None
        self._stream_logger_thread: threading.Thread | None = None
        super().__init__(caller)

    def _setup(self) -> None:
        self._popen = subprocess.Popen(
            self._env.cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        LOGGER.info(f"start Recorder: {self._env.name}")

        self._stream_logger_thread = threading.Thread(
            target=self._stream_logger, args=(self._popen, self._env.name), daemon=True
        )
        self._stream_logger_thread.start()

    def _loop(self) -> None:
        if (self._popen is not None) and (self._popen.poll() is None):
            if self.elapsed >= C.RECORDER_WD_PERIOD:
                if not is_recording(
                    self._env.vfile_name_pat, self._env.video_dir, self._env.obs_time
                ):
                    LOGGER.error(f"KILL INACTIVE STREAM: {self._env.name}")
                    self.send_stop()

                self.reset_elapsed()
        else:
            self.send_stop()

    def _teardown(self) -> None:
        returncode: int | None = None

        if self._popen is not None:
            try:
                self._popen.terminate()
                returncode = self._popen.wait(timeout=C.RECBIN_TERM_TIMEOUT)
            except subprocess.TimeoutExpired:
                LOGGER.warning(f"Could Not Terminate, Killing: {self._env.name}")
                self._popen.kill()
                returncode = self._popen.wait()
        LOGGER.info(f"stop Recorder: {self._env.name}({returncode})")

        if self._stream_logger_thread is not None:
            self._stream_logger_thread.join()

    def _stream_logger(self, popen: subprocess.Popen[str], stream_name: str) -> None:
        LOGGER.info(f"start stream logger: {stream_name}")

        while (popen.stderr is not None) and (popen.poll() is None):
            STREAM_LOGGER.warning(f"[{stream_name}] {popen.stderr.readline().rstrip()}")

        LOGGER.info(f"stop stream logger: {stream_name}")

    @property
    def env(self) -> RecorderEnv:
        return self._env


LooperT = typing.TypeVar("LooperT", bound=Looper)


def filename(*parts: str, ext: str | None = None) -> str:
    filename: str = C.EMPTY_STR.join(parts)

    if (ext is not None) and (len(ext) >= 1):
        filename += "." + ext

    return filename


def read_text(file: Path) -> str | None:
    text: str | None = None

    try:
        text = file.read_text()
    except Exception as e:
        LOGGER.error(f"FAILED TO READ: {file} - {e}")

    return text


def write_text(file: Path, text: str) -> bool:
    is_success: bool = False

    try:
        is_success = file.write_text(text) == len(text)
    except Exception as e:
        LOGGER.error(f"FAILED TO WRITE: {file} - {e}")

    return is_success


def remove(path: Path) -> bool:
    is_success: bool = False

    try:
        if path.is_file():
            path.unlink()
            is_success = True
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            is_success = True
        else:
            LOGGER.error(f"UNSUPPORTED PATH: {path}")
    except Exception as e:
        LOGGER.error(f"FAILED TO REMOVE FILE: {path} - {e}")

    return is_success


def mkdir(dir: Path) -> bool:
    is_success: bool = False

    if dir.is_dir():
        is_success = True
    else:
        try:
            dir.mkdir(parents=True, exist_ok=True)
            is_success = True
        except Exception as e:
            LOGGER.error(f"FAILED TO MAKE DIRECTORY: {dir} - {e}")

    return is_success


def move_file(src: Path | None, dst: Path | None) -> bool:
    is_success: bool = False

    if (src is not None) and (dst is not None) and src.is_file() and mkdir(dst.parent):
        is_success = dst == shutil.move(src, dst)

    return is_success


FindType = typing.Literal["f", "d", "fd"]
FindSort = typing.Literal["n", "t"]


def find(
    pat: str,
    parent: Path,
    mtime: float | None = None,
    type: FindType = "f",
    sort: FindSort = "n",
) -> list[Path]:
    def _bool(_path: Path) -> bool:
        return bool(_path)

    def _is_file(_path: Path) -> bool:
        return _path.is_file()

    def _is_dir(_path: Path) -> bool:
        return _path.is_dir()

    def _is_mtime(_basetime: float, _mtime: float, _path: Path) -> bool:
        return ((diff := (_basetime - _path.stat().st_mtime)) >= 0) and (
            ((_mtime >= 0) and (diff >= _mtime)) or ((_mtime < 0) and (diff <= -_mtime))
        )

    def _mtime_sort(_paths: list[Path]) -> list[Path]:
        return sorted(_paths, key=lambda _path: _path.stat().st_mtime)

    is_type: typing.Callable[[Path], bool] = _bool
    if type == "f":
        is_type = _is_file
    elif type == "d":
        is_type = _is_dir

    is_time: typing.Callable[[Path], bool] = _bool
    if mtime is not None:
        is_time = functools.partial(_is_mtime, time.time(), mtime)

    do_sort: typing.Callable[[list[Path]], list[Path]] = sorted
    if sort == "t":
        do_sort = _mtime_sort

    return do_sort(
        [path for path in parent.glob(pat) if (is_type(path) and is_time(path))]
    )


def usage_rate(dir: Path) -> int | None:
    usage_rate: int | None = None

    try:
        total, used, _ = shutil.disk_usage(dir)
        if (total > 0) and (used >= 0):
            usage_rate = int((used * 100) / total)
    except Exception as e:
        LOGGER.error(f"FAILED TO GET DISK USAGE: {dir} - {e}")

    return usage_rate


def is_recording(vfile_name_pat: str, video_dir: Path, observation_time: float) -> bool:
    return bool(find(vfile_name_pat, video_dir, -observation_time))


def start_looper(looper: LooperT) -> LooperT:
    looper.start()
    return looper


def stop_loopers(loopers: list[LooperT] | list[LooperT | None]) -> None:
    stopping_loopers: list[LooperT] = []
    for looper in loopers:
        if looper is not None:
            looper.send_stop()
            stopping_loopers.append(looper)
    for looper in stopping_loopers:
        try:
            looper.join()
        except Exception:
            pass


def is_alive(looper: Looper | None) -> bool:
    return looper.is_alive() if (looper is not None) else False


def get_alive_loopers(loopers: list[LooperT]) -> list[LooperT]:
    return [looper for looper in loopers if looper.is_alive()]


def get_unreqd_recorders(
    recorders: list[Recorder], recorder_envs: list[RecorderEnv]
) -> list[Recorder]:
    unreqd_recorders_dict: dict[int, Recorder] = {}

    for i in range(len(recorders)):
        if i not in unreqd_recorders_dict:
            if recorders[i].env not in recorder_envs:
                unreqd_recorders_dict[i] = recorders[i]
            else:
                duplicated_dict: dict[int, Recorder] = {
                    i + j: recorders[i + j]
                    for j in range(len(recorders[i:]))
                    if recorders[i + j].env == recorders[i].env
                }
                if len(duplicated_dict) > 1:
                    unreqd_recorders_dict |= duplicated_dict

    return list(unreqd_recorders_dict.values())


def get_reqd_recorder_envs(
    recorder_envs: list[RecorderEnv], recorders: list[Recorder]
) -> list[RecorderEnv]:
    recording_envs: list[RecorderEnv] = [
        recorder.env for recorder in recorders if recorder
    ]
    return [
        recorder_env
        for recorder_env in recorder_envs
        if recorder_env not in recording_envs
    ]


def status(env: Env, lock: Lock) -> int:
    e_code: int = E.NONE

    print(datetime.now().strftime(C.DATE_FORMAT))
    print(C.EMPTY_STR)

    if isinstance((result := lock.check()), bool) and result:
        print("[STREAM]")
        for renv in env.recorder_envs:
            if is_recording(renv.vfile_name_pat, renv.video_dir, renv.obs_time):
                print(f"{renv.name} : OK")
            else:
                print(f"{renv.name} : NG")
                e_code |= E.STREAM
    elif isinstance(result, bool) and not result:
        print("NOT RUNNING")
        e_code |= E.GENERAL
    else:
        print("STATUS UNKNOWN")
        e_code |= E.GENERAL
    print(C.EMPTY_STR)

    print("[STORAGE]")
    if isinstance((current := usage_rate(env.storager_env.video_dir)), int):
        print(f"Current : {current}%")
        print(f"Remove start : {env.storager_env.remove_start}%")
        print(f"Remove stop : {env.storager_env.remove_stop}%")
    else:
        print("DISK USAGE UNKNOWN")
        LOGGER.error("DISK USAGE UNKNOWN")
        e_code |= E.GENERAL
    print(C.EMPTY_STR)

    return e_code


def start(env: Env, lock: Lock) -> int:
    e_code: int = E.NONE

    LOGGER.info("in")

    main_thread = threading.main_thread()
    storager: Storager | None = None
    recorders: list[Recorder] = []
    pid: int = os.getpid()

    if lock.acquire(pid):
        while lock.update(pid):
            if not is_alive(storager):
                storager = start_looper(Storager(main_thread, env.storager_env))

            stop_loopers(get_unreqd_recorders(recorders, env.recorder_envs))
            recorders = get_alive_loopers(recorders)
            for recorder_env in get_reqd_recorder_envs(env.recorder_envs, recorders):
                recorders.append(start_looper(Recorder(main_thread, recorder_env)))

            if not (len(recorders) > 0):
                LOGGER.warning("No Stream To Record")
                e_code |= E.STREAM
                break

            time.sleep(C.MAIN_THREAD_SLEEP)

        stop_loopers(recorders + [storager])
        if not lock.release():
            LOGGER.error("FAILED TO RELEASE LOCK")

    else:
        LOGGER.warning("Already Running")
        e_code |= E.GENERAL

    LOGGER.info(f"out: {e_code}")

    return e_code


def stop(env: Env, lock: Lock) -> int:
    e_code: int = E.NONE

    LOGGER.info("in")

    if isinstance(result := lock.stop(), bool) and not result:
        LOGGER.error("FAILED TO STOP")
        e_code |= E.GENERAL
    elif not isinstance(result, bool):
        LOGGER.warning("Not Running")
        e_code |= E.GENERAL

    LOGGER.info(f"out: {e_code}")

    return e_code


def init(cwd: Path) -> tuple[Env | None, Lock | None]:
    env: Env | None = None
    lock: Lock | None = None

    is_dir_set: bool = True
    try:
        env = Env(cwd)
        for dir in [env.log_env.log_dir, env.storager_env.archive_dir]:
            is_dir_set &= mkdir(dir)
    except Exception as e:
        is_dir_set = False
        print(f"FAILED TO INITIALIZE: {cwd} - {e}", file=sys.stderr)

    if (env is not None) and is_dir_set:
        lock = Lock(cwd)

        handler = TimedRotatingFileHandler(
            env.log_env.log_dir / C.LOG_NAME,
            when=C.LOG_WHEN,
            backupCount=env.log_env.log_backup,
            interval=C.LOG_INTERVAL,
            encoding=C.LOG_ENCODING,
        )
        handler.setFormatter(logging.Formatter(C.LOG_FORMATTER))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(C.LOG_LEVEL)

        stream_handler = RotatingFileHandler(
            env.log_env.log_dir / C.STREAMLOG_NAME,
            maxBytes=env.log_env.streamlog_size,
            backupCount=env.log_env.streamlog_backup,
            encoding=C.STREAMLOG_ENCODING,
        )
        stream_handler.setFormatter(logging.Formatter(C.STREAMLOG_FORMATTER))
        STREAM_LOGGER.addHandler(stream_handler)
        STREAM_LOGGER.setLevel(C.STREAMLOG_LEVEL)

    else:
        env = None
        print("FAILED TO PREPARE DIRECTORIES", file=sys.stderr)

    return (env, lock)


def main() -> int:
    e_code: int = E.NONE

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.set_defaults(func=status)
    parser.add_argument("-v", "--version", action="version", version=C.VERSION)
    parser.add_argument("-d", "--dir", default=os.getcwd())
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser(C.SUBCOMMAND_START)
    subparser.set_defaults(func=start)

    subparser = subparsers.add_parser(C.SUBCOMMAND_STOP)
    subparser.set_defaults(func=stop)

    args: argparse.Namespace = parser.parse_args()
    env, lock = init(Path(args.dir).resolve())
    if isinstance(env, Env) and isinstance(lock, Lock):
        try:
            e_code |= args.func(env, lock)
        except Exception as e:
            LOGGER.critical(f"UNEXPECTED ERROR: {e}")
            e_code |= E.GENERAL
    else:
        e_code |= E.PREF

    return e_code


if __name__ == "__main__":
    sys.exit(main())
