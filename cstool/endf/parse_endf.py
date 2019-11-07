#!/usr/bin/env python3
# Use on
# EADL (Evaluated Atomic Data Library) and
# EEDL (Evaluated Electron Data Library) files
# in ENDF (Evaluated Nuclear Data File) format
# http://www.nndc.bnl.gov/endf/
# ENDF/B-VII.1 (2012):
# http://www.nndc.bnl.gov/endf/b7.1/download.html

import re
from io import StringIO
from zipfile import ZipFile
from .endf_reader import endf_data

def parse_photoat(filename, Z):
    with ZipFile(filename) as zf:
        pattern = re.compile(
            '^photoat/photoat-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
        for fn in zf.namelist():
            if pattern.match(fn):
                return endf_data(StringIO(zf.read(fn).decode('utf-8')))

def parse_atomic_relax(filename, Z):
    with ZipFile(filename) as zf:
        pattern = re.compile(
            '^atomic_relax/atom-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
        for fn in zf.namelist():
            if pattern.match(fn):
                return endf_data(StringIO(zf.read(fn).decode('utf-8')))

def parse_electrons(filename, Z):
    with ZipFile(filename) as zf:
        pattern = re.compile(
            '^electrons/e-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
        for fn in zf.namelist():
            if pattern.match(fn):
                return endf_data(StringIO(zf.read(fn).decode('utf-8')))
