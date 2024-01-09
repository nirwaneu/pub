#!/bin/bash
rm /usr/bin/apktool* 1>/dev/null
wget https://raw.githubusercontent.com/iBotPeaches/Apktool/master/scripts/linux/apktool -O /usr/bin/apktool -q --show-progress
wget https://bitbucket.org/iBotPeaches/apktool/downloads/apktool_2.9.0.jar -O /usr/bin/apktool.jar -q --show-progress
chmod +x /usr/bin/apktool*
