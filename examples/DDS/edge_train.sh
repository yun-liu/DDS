#!/bin/bash

set -x

LOG="logs/edge_sbd_`date +%Y-%m-%d_%H-%M-%S`.log"
exec &> >(tee -a "$LOG")

/usr/bin/python edge_solve.py
