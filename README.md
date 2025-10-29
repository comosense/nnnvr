# nnnvr
**nnnvr (No-Named Network Video Recorder):** A simple, minimalist Network Video Recording (NVR) solution.

## Prerequisites
While `nnnvr` has only been tested in a limited environment[^1], it is designed to run on most systems that meet the following requirements.

**Note:** The following examples are tailored for Debian-based Linux. Please adapt these commands for your specific operating systems (e.g., using `PowerShell`, `Task Scheduler` on Windows, etc.).


### IP Cameras
You'll need IP cameras that support **RTSP** streaming.

### ffmpeg
Install [`ffmpeg`](https://www.ffmpeg.org/) if you don't already have it. On Debian / Ubuntu, the typical installation commands are:
```sh
sudo apt-get update
sudo apt install ffmpeg
```

### python3
Install [`python3`](https://www.python.org/) if you don't already have it. On Debian / Ubuntu, the typical installation commands are:
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
  Replace [VERSION] with the latest version tag from the releases page (e.g., 1.0.12). See: https://github.com/comosense/nnnvr/releases

### 2. `nnnvr.json`
Create a configuration file named `nnnvr.json` in the same working directory. Adjust the settings to match your environment.

**SECURITY WARNING:** This file contains credentials (RTSP usernames and passwords). It is **crucial** to restrict its permissions, such as `chmod 600 ./nnnvr.json`.

* Example 1: Minimal `nnnvr.json`
  ```JSON
  {
      "streams":
      [
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
      "streams":
      [
          {
              "name": "cctv-X",
              "url": "rtsp://USER_X:PASS_X@XXX.XXX.XXX.XXX:XXX/streamX",
              "transport": "tcp",
              "ext": "mp4",
              "vcodec": "libx264",
              "fps": 30,
              "acodec": "aac",
              "segmentSec": 600,
              "obsSec": 120
          },
          {
              "name": "cctv-Y",
              "url": "rtsp://USER_Y:PASS_Y@YYY.YYY.YYY.YYY:YYY/streamY"
          },
          {
              "name": "cctv-Z",
              "url": "rtsp://USER_Z:PASS_Z@ZZZ.ZZZ.ZZZ.ZZZ:ZZZ/streamZ",
              "transport": "udp",
              "ext": "ts",
              "vcodec": "copy",
              "acodec": "copy",
              "obsSec": 600
          }
      ],
      "recBin": "/PATH/TO/ffmpeg",
      "log":
      {
          "dir": "/PATH/TO/log",
          "logBackup": 14,
          "streamlogSizeKb": 200,
          "streamlogBackup": 3
      },
      "video":
      {
          "dir": "/PATH/TO/video",
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
|`log`|No|JSON (See **Log Configuration** below)|Preferences for log file management.|(See **Log Configuration** below)|
|`video`|No|JSON (See **Video Configuration** below)|Preferences for video storage management.|(See **Video Configuration** below)|

#### Stream Configuration ("stream" JSON)
|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`name`|Yes|String|A unique name for the stream (e.g., `cctv-X`). **Must be unique.** across all defined streams.|-|
|`url`|Yes|String|RTSP URL (e.g., `"rtsp://..."`).|-|
|`transport`|No|String|RTSP transport protocol (Equivalent to `ffmpeg`'s `-rtsp_transport` option. Common values are `"udp"` or `"tcp"`).|`"udp"`|
|`ext`|No|String|Video file extension and container format (e.g., `"mp4"`, `"ts"`).|`"mp4"`|
|`vcodec`|No|String|Video codec for recording (Equivalent to `ffmpeg`'s `-c:v` option).|(Depends on `ffmpeg`)|
|`fps`|No|Integer|Frames per second for recording (Equivalent to `ffmpeg`'s `-r` option).|(Depends on `ffmpeg`)|
|`acodec`|No|String|Audio codec for recording (Equivalent to `ffmpeg`'s `-c:a` option).|(Depends on `ffmpeg`)|
|`segmentSec`|No|Integer|Duration (in seconds) for splitting the recorded video files (Equivalent to `ffmpeg`'s `-segment_time` option).|`900`|
|`obsSec`|No|Integer|Observation window (in seconds). If no video file is updated within this period, the recorder will restart the stream.|`segmentSec`+`60`|

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
|`removeStart`|No|Integer|Disk usage percentage (1-99) to **trigger** removal of old archives.|`99`|
|`removeStop`|No|Integer|Disk usage percentage (1-99) to **stop** removal. **Must be $\le$ `removeStart`**.|`99`|

## Usage and Testing
**Important:** Ensure the user executing these commands has the necessary permissions to run `ffmpeg`.

### 1. Start `nnnvr`
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

### 3. Stop `nnnvr`
```sh
./nnnvr.py stop
```

## Deployment
For persistent background operation, running `nnnvr` as a `systemd` service is recommended.
Create the file `nnnvr.service` based on your environment.
* Example: `nnnvr.service`:
  ```sh
  [Unit]
  Description=No-Named NVR (nnnvr) Service
  After=network.target

  [Service]
  Type=simple
  Restart=always
  WorkingDirectory=/WORK/DIR
  ExecStart=/PATH/TO/nnnvr.py start
  ExecStop=/PATH/TO/nnnvr.py stop

  [Install]
  WantedBy=multi-user.target

Place `nnnvr.service` in the systemd directory (e.g., `/etc/systemd/system/`) and then enable and start the service:
```sh
sudo mv ./nnnvr.service /etc/systemd/system/.
sudo systemctl daemon-reload
sudo systemctl enable nnnvr
sudo systemctl start nnnvr
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[^1]: [`dietpi(v9.17.2)`](https://dietpi.com/), [`python(3.13.5)`](https://www.python.org/), [`ffmpeg-rockchip`](https://github.com/nyanmisaka/ffmpeg-rockchip), [`Radxa ZERO 3E`](https://radxa.com/products/zeros/zero3e) and [`C530WS`](https://www.tp-link.com/en/home-networking/cloud-camera/tapo-c530ws/)
