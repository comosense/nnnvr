# nnnvr

**nnnvr (No-Named Network Video Recorder):** A simple, minimalist Network Video Recording (NVR) solution.

## Prerequisites

While **nnnvr** has only been tested in a limited environment[^1], it is designed to run on most systems that meet the following requirements.

**Note:** The following examples are tailored for Debian-based Linux. Please adapt these commands for your specific operating systems (e.g., using **PowerShell**, **Task Scheduler** on Windows, etc.).

### IP Cameras

You'll need IP cameras that support **RTSP** streaming.

### FFmpeg

**nnnvr** uses **FFmpeg** as an external command-line tool.
**FFmpeg** is **not included** in this repository. It is developed and licensed independently under the [LGPL or GPL](https://ffmpeg.org/legal.html) by the [FFmpeg project](https://www.ffmpeg.org/).
**Important:** Available container formats and audio/video codecs depend on your system. **nnnvr** does not include any container formats, codecs, or patent licenses.

Install **FFmpeg** if you don't already have it. On Debian / Ubuntu, the typical installation commands are:

```sh
sudo apt-get update
sudo apt install ffmpeg
```

### Python3

**nnnvr** is written in **Python**.
**Python** is developed and licensed independently under the [Python Software Foundation License](https://docs.python.org/3/license.html) by the [Python Software Foundation](https://www.python.org/).

Install **Python3** if you don't already have it. On Debian / Ubuntu, the typical installation commands are:

```sh
sudo apt-get update
sudo apt install python3
```

## Installation

### 1. `nnnvr.py`

Place `nnnvr.py` in your desired working directory and make it executable.

* Example on Debian / Ubuntu:

    ```sh
    cd /WORK/DIR
    curl https://raw.githubusercontent.com/comosense/nnnvr/refs/tags/[VERSION]/nnnvr.py > nnnvr.py
    chmod +x ./nnnvr.py
    ```

    Replace [VERSION] with the latest version tag from the releases page (e.g., 1.1.1). See: <https://github.com/comosense/nnnvr/releases>

### 2. `nnnvr.json`

Create a configuration file named `nnnvr.json` in the same working directory. Adjust the settings to match your environment.

**SECURITY WARNING:** This file contains credentials (RTSP usernames and passwords). It is **crucial** to restrict its permissions, such as `chmod 600 ./nnnvr.json`.

* Example 1: Minimal `nnnvr.json`

    ```JSON
    {
        "streams": [
            {
                "name": "cctv",
                "url": "rtsp://USER:PASS@XXX.XXX.XXX.XXX:XXX/streamX"
            }
        ]
    }
    ```

* Example 2: Comprehensive `nnnvr.json`

    ```JSON
    {
        "streams": [
            {
                "name": "cctv-X",
                "url": "rtsp://USER_X:PASS_X@XXX.XXX.XXX.XXX:XXX/streamX",
                "transport": "udp",
                "ext": "avi",
                "vcodec": "mjpeg",
                "fps": 30,
                "acodec": "pcm_alaw",
                "segmentSec": 600,
                "obsSec": 120,
                "oOptions": [
                    "-vf",
                    "scale=640:360"
                ]
            },
            {
                "name": "cctv-Y",
                "url": "rtsp://USER_Y:PASS_Y@YYY.YYY.YYY.YYY:YYY/streamY"
            },
            {
                "name": "cctv-Z",
                "url": "rtsp://USER_Z:PASS_Z@ZZZ.ZZZ.ZZZ.ZZZ:ZZZ/streamZ",
                "transport": "tcp",
                "ext": "mkv",
                "vcodec": "copy",
                "acodec": "copy",
                "segmentSec": 900,
                "obsSec": 600,
                "gOptions": [
                    "-report"
                ],
                "iOptions": [
                    "-use_wallclock_as_timestamps",
                    "1"
                ]
            }
        ],
        "recBin": "/PATH/TO/ffmpeg",
        "tempDir": "PATH/TO/TEMP_DIR",
        "log": {
            "dir": "/PATH/TO/LOG_DIR",
            "logBackup": 14,
            "streamlogSizeKb": 200,
            "streamlogBackup": 3
        },
        "video": {
            "dir": "/PATH/TO/VIDEO_DIR",
            "archivingWaitHour": 3,
            "removeStart": 90,
            "removeStop": 80
        }
    }
    ```

#### Top-Level Configuration

|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`streams`|Yes|JSON array (See **Stream Configuration** below)|An array of IP camera stream configuration objects.|-|
|`recBin`|No|String|Path to the `ffmpeg` executable.|`"ffmpeg"`|
|`tempDir`|No|String|Path to the temporary directory used for the lock file. If not specified, the system's default temporary directory is used.|`<system_temp_dir>/nnnvr`|
|`log`|No|JSON (See **Log Configuration** below)|Preferences for log file management.|(See **Log Configuration** below)|
|`video`|No|JSON (See **Video Configuration** below)|Preferences for video storage management.|(See **Video Configuration** below)|

#### Stream Configuration ("stream" JSON)

|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`name`|Yes|String|A unique name for the stream (e.g., `cctv-X`). **Must be unique** across all defined streams.|-|
|`url`|Yes|String|RTSP URL (e.g., `"rtsp://..."`).|-|
|`transport`|No|String|RTSP transport protocol (Equivalent to `ffmpeg`'s `-rtsp_transport` option. Common values are `"udp"` or `"tcp"`).|`"udp"`|
|`ext`|No|String|Video file extension and container format (e.g., `"mkv"`, `"avi"`).|`"mkv"`|
|`vcodec`|No|String|Video codec for recording (Equivalent to `ffmpeg`'s `-c:v` option).|(Depends on `ffmpeg`)|
|`fps`|No|Integer|Frames per second for recording (Equivalent to `ffmpeg`'s `-r` option).|(Depends on `ffmpeg`)|
|`acodec`|No|String|Audio codec for recording (Equivalent to `ffmpeg`'s `-c:a` option).|(Depends on `ffmpeg`)|
|`segmentSec`|No|Integer|Duration (in seconds) for splitting the recorded video files (Equivalent to `ffmpeg`'s `-segment_time` option).|`900`|
|`obsSec`|No|Integer|Observation window (in seconds). If no video file is updated within this period, the recorder will restart the stream.|`segmentSec`+`60`|
|`gOptions`|No|JSON array|An array of strings containing `ffmpeg` global options. (e.g., `["-report"]`)|-|
|`iOptions`|No|JSON array|An array of strings containing `ffmpeg` input options. (e.g., `["-use_wallclock_as_timestamps", "1"]`)|-|
|`oOptions`|No|JSON array|An array of strings containing `ffmpeg` output options. (e.g., `["-vf", "scale=640:360"]`)|-|

**Important:** `gOptions`, `iOptions`, and `oOptions` allow you to pass arbitrary global, input, and output options directly to the underlying **ffmpeg** process.

* **Requires FFmpeg Knowledge:** Because these options are injected directly into the command line, you must have a thorough understanding of **FFmpeg**'s specifications and argument syntax.
* **Avoid Conflicts:** Be extremely careful not to duplicate or conflict with options already generated by nnnvr's standard Stream Configuration (such as `vcodec`, `fps`, `transport`, `segmentSec`, etc.). Conflicting or incompatible arguments will cause the stream recording to fail or terminate unexpectedly.

#### Log Configuration ("log" JSON)

|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`dir`|No|String|Path to the log directory.|`"<working directory>/log"`|
|`logBackup`|No|Integer|Number of daily log files to retain.|`28`|
|`streamlogSizeKb`|No|Integer|Maximum size (in KBytes) of a stream log file.|`100`|
|`streamlogBackup`|No|Integer|Number of stream log files to retain. Logs are rotated when they reach `streamlogSizeKb`.|`5`|

#### Video Storage Configuration ("video" JSON)

|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`dir`|No|String|Path to the video directory.|`"<working directory>/video"`|
|`archivingWaitHour`|No|Integer|Wait time (in hours) before archiving each video file.|`6`|
|`removeStart`|No|Integer|Disk usage percentage (1-99) to **trigger** removal of old archives.|`90`|
|`removeStop`|No|Integer|Disk usage percentage (1-99) to **stop** removal. **Must be <= `removeStart`**.|`removeStart`|

## Usage and Testing

**Important:** Ensure the user executing these commands has the necessary permissions to run `ffmpeg`.

### 1. Start

```sh
cd /WORK/DIR
./nnnvr.py start &
```

If successful, video files will appear in the specified video directory. If not, check the log files for errors.

If `nnnvr.py` and `nnnvr.json` are not in the same directory, use the `-d` option, like `/PATH/TO/nnnvr.py -d /WORK/DIR start &`. **This applies to all commands(`start`, `stop` and status).**

### 2. Check Status

```sh
./nnnvr.py
```

This command shows the current `nnnvr` status (e.g., recording activity, disk usage).

### 3. Stop

```sh
./nnnvr.py stop
```

## Deployment

For persistent background operation, running `nnnvr` as a `systemd` service is recommended.
Create the file `nnnvr.service` based on your environment.

* Example: `nnnvr.service`:

    ```ini
    [Unit]
    Description=No-Named NVR (nnnvr) Service
    After=network.target

    [Service]
    Type=simple
    Restart=always
    RestartSec=10s
    StartLimitInterval=60s
    StartLimitBurst=5
    WorkingDirectory=/WORK/DIR
    ExecStart=/PATH/TO/nnnvr.py start
    ExecStop=/PATH/TO/nnnvr.py stop

    [Install]
    WantedBy=multi-user.target
    ```

Place `nnnvr.service` in the systemd directory (e.g., `/etc/systemd/system/`) and then enable and start the service:

```sh
sudo mv ./nnnvr.service /etc/systemd/system/.
sudo systemctl daemon-reload
sudo systemctl enable nnnvr
sudo systemctl start nnnvr
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

As **nnnvr** is designed for 24/7 video recording, please be aware that the authors are not responsible for any data loss, hardware wear (such as SD card or SSD/HDD failure), or missed recordings due to software bugs, network issues, or power failures. Users are encouraged to verify the stability of their own setup.

**nnnvr** does **not** bundle, distribute, or install **FFmpeg**.
It simply invokes a **FFmpeg** executable that is installed separately by the user.
Available container formats and audio/video codecs depend on the **FFmpeg** build you install, your operating system and hardware, and the laws and license terms applicable in your jurisdiction.
You are responsible for selecting and using containers and codecs appropriately for your environment and for ensuring compliance with any applicable licenses, patents, and legal requirements.
**nnnvr** does **not** grant, provide, or sublicense any patent, container format, or codec licenses.

[^1]: [`dietpi(v9.17.2)`](https://dietpi.com/), [`python(3.13.5)`](https://www.python.org/), [`ffmpeg-rockchip`](https://github.com/nyanmisaka/ffmpeg-rockchip), [`Radxa ZERO 3E`](https://radxa.com/products/zeros/zero3e) and [`C530WS`](https://www.tp-link.com/en/home-networking/cloud-camera/tapo-c530ws/)
