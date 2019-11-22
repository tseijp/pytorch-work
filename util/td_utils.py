import os
import sys
import time
import socket
import argparse
import subprocess

DIRECTORY = r'C:\Users\yousei\dir_jk\pytorch-yanai'
#os.path.dirname(os.path.abspath('__file__'))
class Socket:
    def __init__(self,opt):
        self.port= opt.port if opt.port else 7001
        self.cmd = opt.cmd  if opt.cmd  else "python --version"
        self.msg = opt.msg.strip('"')  if opt.msg  else "test"
        self.dir = opt.dir.strip('"').strip(' ')  if opt.dir  else DIRECTORY
        self.ip  = opt.ip .strip('"').strip(' ')  if opt.ip   else "127.0.0.1"
        self.s  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sub
        self.sleep=float(opt.sleep) if opt.sleep else 0

        if 1:#msg:
            self.send_msg(self.dir)
            self.send_msg('port:%s,cmd:%s,ip:%s'%(self.port,self.cmd,self.ip))
    def send_msg(self, msg='no message'):
        text = msg.encode('utf-8') if type(msg) is str else msg
        print(text)
        self.s.sendto(text, (self.ip, int(self.port)))

    def run(self):
        self.send_msg('/%sStarted%s/'%('*'*20,'*'*20))
        if os.path.isdir(self.dir) is False:
            self.send_msg("%s"%os.path.isdir(self.dir))
            self.send_msg("Error : there is no dir %s"%self.dir)
            return
        # main process ---------------------------------------------------------
        try:
            self.send_msg('%s'%self.cmd)
            sub = subprocess.Popen(self.cmd,#cmd_list,
                cwd=self.dir, shell=True,
                #stdout = subprocess.PIPE,
                #stderr = subprocess.PIPE,
                universal_newlines=True,
                bufsize=0)#[ref](https://janakiev.com/blog/python-shell-commands/)

            if self.sleep>0:
                time.sleep(self.sleep)

            # good 1---------------------
            #for line in iter(proc.stdout.readline, b''):
            #    print(line.rstrip())
            # good 2---------------------
            #while True:
                #line = proc.stdout.readline()
                #if line:
                #    self.send_msg(line)
                #if not line and proc.poll() is not None:
                #    break
            # good 3---------------------
            #while True:
            #    output = process.stdout.readline()
            #    print(output.strip())
            #    return_code = process.poll()                # Do something else
            #    if return_code is not None:
            #        print('RETURN CODE', return_code)
            #        for output in process.stdout.readlines():
            #            print(output.strip()) # Process has finished, read rest of the output
            #        break
        except:
            import traceback
            self.send_msg('[Error] in main process of Socket.run():%s'%traceback.format_exc())
        # ----------------------------------------------------------------------
        self.send_msg('/%sFinished%s/'%('*'*20,'*'*20))

class SocketOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument('-p', '--port' ,type=str ,default='')
        self.parser.add_argument('-c', '--cmd'  ,type=str ,default='')
        self.parser.add_argument('-m', '--msg'  ,type=str ,default='')
        self.parser.add_argument('-d', '--dir'  ,type=str ,default='')
        self.parser.add_argument('-i', '--ip'   ,type=str ,default='')
        # sub
        self.parser.add_argument('-s', '--sleep',type=str ,default='')

    def parse(self, save=True):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

def main():
    opt = SocketOption().parse()
    soc = Socket(opt)
    soc.run()

if __name__=='__main__':
    main()
