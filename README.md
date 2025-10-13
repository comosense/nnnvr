# nnnvr
**nnnvr (No-Named Network Video Recorder):** A straightforward, minimalist Network Video Recording solution.

## Prerequisites
While `nnnvr` has only been tested in a limited environment[^1], it is designed to function on most systems that meet the following requirements, regardless of the specific operating system or hardware.

**Note:** The following examples are tailored for Debian-based Linux. Please adapt them for other operating systems (e.g., using `PowerShell`, `Task Scheduler` on Windows, etc.).


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
Place `nnnvr.py` in your desired working directory and grant it executable permission.
* Example on Debian / Ubuntu:
  ```sh
  cd /WORK/DIR
  curl https://raw.githubusercontent.com/comosense/nnnvr/refs/tags/[VERSION]/nnnvr.py > nnnvr.py
  chmod +x ./nnnvr.py
  ```

### 2. `nnnvr.json`
Create the configuration file, `nnnvr.json`, in the working directory, and adjust the settings to match your environment.

**SECURITY WARNING:** Since `nnnvr.json` contains the RTSP URL (including user/password), it is **crucial** to set appropriate file permissions, such as `chmod 600 ./nnnvr.json`.

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
              "ext": "mp4",
              "vcodec": "libx264",
              "fps": 30,
              "acodec": "aac",
              "segmentSec": 600
          },
          {
              "name": "cctv-Y",
              "url": "rtsp://USER_Y:PASS_Y@YYY.YYY.YYY.YYY:YYY/streamY"
          },
          {
              "name": "cctv-Z",
              "url": "rtsp://USER_Z:PASS_Z@ZZZ.ZZZ.ZZZ.ZZZ:ZZZ/streamZ",
              "ext": "ts",
              "vcodec": "copy",
              "acodec": "copy"
          }
      ],
      "recBin": "/PATH/TO/ffmpeg",
      "log":
      {
          "dir": "/PATH/TO/log",
          "logBackup": 14,
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
|`recBin`|No|String|The specified path to the `ffmpeg` executable.|`"ffmpeg"`|
|`log`|No|JSON (See **Log Configuration** below)|Preferences for log file management.|(See **Log Configuration** below)|
|`video`|No|JSON (See **Video Configuration** below)|Preferences for video storage management.|(See **Video Configuration** below)|

#### Stream Configuration ("stream" JSON)
|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`name`|Yes|String|A unique name for the stream (e.g., `cctv-X`). **Must be unique.**|-|
|`url`|Yes|String|The RTSP URL (e.g., `"rtsp://..."`).|-|
|`ext`|No|String|Video file extension and container format (e.g., `mp4`, `ts`).|`"mp4"`|
|`vcodec`|No|String|Video codec for recording (Equivalent to `ffmpeg`'s `-c:v` option).|(Depends on `ffmpeg`)|
|`fps`|No|Integer|Frames per second for recording (Equivalent to `ffmpeg`'s `-r` option).|(Depends on `ffmpeg`)|
|`acodec`|No|String|Audio codec for recording (Equivalent to `ffmpeg`'s `-c:a` option).|(Depends on `ffmpeg`)|
|`segmentSec`|No|Integer|Duration (in seconds) for splitting the recorded video files (Equivalent to `ffmpeg`'s `-segment_time` option).|`900`|

#### Log Configuration ("log" JSON)
|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`dir`|No|String|The specified path to the directory for log files.|`"(working directory)/log"`|
|`logBackup`|No|Integer|The number of main daily log files to keep.|`28`|
|`streamlogBackup`|No|Integer|The number of stream log files to keep (Stream logs roll over at 100KB).|`5`|

#### Video Storage Configuration ("video" JSON)
|Key|Required|Type|Description|Default|
|:-|:-|:-|:-|:-|
|`dir`|No|String|The specified path to the directory for video files.|`"(working directory)/video"`|
|`archivingWaitHour`|No|Integer|Wait time (in hours) before archiving each video file.|`6`|
|`removeStart`|No|Integer|The disk usage percentage threshold to start removing the oldest videos.|`99`|
|`removeStop`|No|Integer|The disk usage percentage threshold to stop removing videos.|`99`|

## Usage and Testing
**Important:** Ensure the user executing these commands has the necessary permissions to run `ffmpeg`.

### 1. Start `nnnvr`
```sh
cd /WORK/DIR
./nnnvr.py start &
```
If successful, a recorded file should be created in the video directory. Check the logs for troubleshooting if it fails.

If `nnnvr.py` and `nnnvr.json` are not in the same directory, use the `-d` option, like `/PATH/TO/nnnvr.py -d /WORK/DIR start &`. **This directory requirement applies to all subsequent commands.**

### 2. Check Status
```sh
./nnnvr.py
```
This command show the current `nnnvr` status (e.g., recording activity, disk usage).

### 3. Stop `nnnvr`
```sh
./nnnvr.py stop
```

## Deployment
The fastest way to run `nnnvr` as a persistent background service is by using `systemd` for daemonization.
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
  ExecReload=/PATH/TO/nnnvr.py restart

  [Install]
  WantedBy=multi-user.target

Place `nnnvr.service` in the systemd directory and start the service:
```sh
sudo mv ./nnnvr.service /etc/systemd/system/.
sudo systemctl daemon-reload
sudo systemctl enable nnnvr
sudo systemctl start nnnvr
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[^1]: [`dietpi(v9.17.2)`](https://dietpi.com/), [`python(3.13.5)`](https://www.python.org/), [`ffmpeg-rockchip`](https://github.com/nyanmisaka/ffmpeg-rockchip), [`Radxa ZERO 3E`](https://radxa.com/products/zeros/zero3e) and [`C530WS`](https://www.tp-link.com/en/home-networking/cloud-camera/tapo-c530ws/)
