# GANtools

Python-based cli tools for generating images using bigGAN.

adjustments to run it GPU:
make conda environment
- install tensorflow-gpu
- install tensorflow-hub
- download biggan from hub and give path to biggan.py
- python3 setup.py install
## Instructions
1. Install Python 3 x64; it is very important that you get the 64 bit version or it will not work. If you are on Windows, be sure to check the "add to PATH" box in the installer. (Note for macOS users: if you are going to use brew to install python make sure you replace `python` with `python3` and `pip` with `pip3` in all of the following steps).
2. Install git
https://git-scm.com/downloads
3. Install gantools
```
pip install git+https://gitlab.com/Vee9ahd1/gantools
```
4. Read the help dialog
```
gantools -h
```
## Example usage
```
gantools --username username@email.com --password mypassword123 --nframes 20 --keys 7968340a72eabab735d04dba 0416461072e5e22fd6d1637c c37d216dfd865aa7397db242
```
where the hexidecimal strings after `--keys` are the IDs found in an image's ganbreeder URL (e.g. https://ganbreeder.app/i?k=c37d216dfd865aa7397db242 has key: c37d216dfd865aa7397db242). You can list arbitrarily many keys; just be sure to increase your `--nframes` value to compensate (and be prepared for a longer render time).
## Troubleshooting
### How do I open the terminal emulator / command prompt?
Windows: press the Windows key and the r key at the same time and then run `cmd.exe` in the dialog that pops up.

Mac: run `Terminal.app` in your `/Applications/Utilities` directory.

Linux: install gentoo.
### Syntax error / command not found / etc. on Windows
Did you check "add Python to PATH" in the installer? Probably not. Reinstall and be sure to do that.
### Missing implementation that supports loader(\*(\'...
Somehow the tfhub cache gets invalidated sometimes. IDK what causes it at that point but if you delete the tfhub\_modules subdirectory it gives you the problem will be fixed (note: next time you run the script it will have to re-download the model so it will be slow).
### SSL missing issuer certificate.
```
"urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1056)>"
```
AFAIK this is a MacOS only issue. Look in the Python3 install folder and you'll see a file called `Install Certificates.command`. Double click to run it. If that doesn't work (or if you installed Python with brew) try this:
```
pip3 install --upgrade certifi
```
