#! /bin/bash


# original: /data/images/
# new: /home/weikefan/helper/office-home

sed -i 's|/home/weikefan/helper/office-home/|/home/weikefan/helper/NCL-reproduce/office-home/|g' ./NCL-changed/data/office-home/*.txt
