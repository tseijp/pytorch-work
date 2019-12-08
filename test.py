import os
import sys
import subprocess

arg0 = '"C:/Program Files/Derivative/TouchDesigner099/bin/python.exe"'
arg1 = 'pose_comp.py'
cmd  = ' '.join( [arg0, arg1] )#+ args)
dir  = r'C:\Users\yousei\dir_jk\pytorch-yanai'
util = 'util/td_utils.py'
proc = subprocess.Popen(['python', util,
	'-p','50001',
	'-d',dir,
	'-c',cmd,
	'-s','15',],
	cwd=dir,
	shell=True)
