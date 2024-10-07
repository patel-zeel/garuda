<!-- align h1 to center -->
<h1 align="center">
    Garuda
</h1>

<p align="center">
  <img src="logo/garuda_profile_full1.png" width="100%">
</p>
<p align="center">
  A research-oriented computer vision library for satellite imagery.
</p>

[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/garuda/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/garuda?branch=main)

## Installation

Stable version:

```bash
pip install garuda
```

Latest version:

```bash
pip install git+https://github.com/patel-zeel/garuda
```

## Usage

See the [examples](examples) directory for more details.

## Functionality

### Configuration

#### Disable/Enable TQDM

```python
from garuda.config import enable_tqdm, disable_tqdm
enable_tqdm()
disable_tqdm()
```

Output:
```python
GARUDA INFO     : TQDM progress bar enabled
GARUDA INFO     : TQDM progress bar disabled
```

#### Adjust log level

```python
from garuda.config import set_log_level
from garuda.base import logger

def _log_everything():
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')
set_log_level('DEBUG')
_log_everything()
set_log_level('INFO')
_log_everything()
set_log_level('WARNING')
_log_everything()
set_log_level('ERROR')
_log_everything()
set_log_level('CRITICAL')
_log_everything()
```

Output:
```python
Log level set to DEBUG
GARUDA DEBUG    : Debug message
GARUDA INFO     : Info message
GARUDA WARNING  : Warning message
GARUDA ERROR    : Error message
GARUDA CRITICAL : Critical message
Log level set to INFO
GARUDA INFO     : Info message
GARUDA WARNING  : Warning message
GARUDA ERROR    : Error message
GARUDA CRITICAL : Critical message
Log level set to WARNING
GARUDA WARNING  : Warning message
GARUDA ERROR    : Error message
GARUDA CRITICAL : Critical message
Log level set to ERROR
GARUDA ERROR    : Error message
GARUDA CRITICAL : Critical message
Log level set to CRITICAL
GARUDA CRITICAL : Critical message
```