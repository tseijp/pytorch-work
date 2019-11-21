import os
import sys
import time
import socket
import argparse
import subprocess

DIRECTORY = os.path.dirname(os.path.abspath('__file__'))
class Socket:
    def __init__(self,opt):
        self.port= opt.port if opt.port else 7001
        self.cmd = opt.cmd  if opt.cmd  else "python --version"
        self.msg = opt.msg.strip('"')  if opt.msg  else "test"
        self.dir = opt.dir.strip('"').strip(' ')  if opt.dir  else DIRECTORY
        self.ip  = opt.ip .strip('"').strip(' ')  if opt.ip   else "127.0.0.1"
        self.s  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if 1:#msg:
            self.send_msg(self.dir)
            self.send_msg('port:%s,cmd:%s,ip:%s'%(self.port,self.cmd,self.ip))
    def send_msg(self, msg='no message'):
        text = msg.encode('utf-8') if type(msg) is str else msg
        print(text)
        self.s.sendto(text, (self.ip, int(self.port)))

    def run(self):
        self.send_msg('Started !----------')
        if os.path.isdir(self.dir) is False:
            self.send_msg("%s"%os.path.isdir(self.dir))
            self.send_msg("Error : there is no dir %s"%self.dir)
            return

        # main process --------------------
        if 1:#try:
            proc = subprocess.Popen(self.cmd,#cmd_list,
                cwd=self.dir, shell=True,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE)

            while True:
                line = proc.stdout.readline()
                if line:
                    self.send_msg(line)
                if not line and proc.poll() is not None:
                    break
        #except:
        #    import traceback
        #    self.send_msg('Error in main process of Socket.run()')
        #    traceback.print_exc()
        # ---------------------------------
        self.send_msg('Finished !---------')

class SocketOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument('-p', '--port' ,type=str ,default='')
        self.parser.add_argument('-c', '--cmd'  ,type=str ,default='')
        self.parser.add_argument('-m', '--msg'  ,type=str ,default='')
        self.parser.add_argument('-d', '--dir'  ,type=str ,default='')
        self.parser.add_argument('-i', '--ip'   ,type=str ,default='')

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
