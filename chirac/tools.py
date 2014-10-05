# -*- coding: utf-8 -*-
def readAFile(nf):
    f = open(nf, 'rb')

    txt = f.readlines()
    txt = ' '.join(txt)

    f.close()
    return txt

def compteLignes(nf, fdl='\n', tbuf=16384):
    """Compte le nombre de lignes du fichier nf"""
    c = 0
    f = open(nf, 'rb')
    while True:
        buf = None
        buf = f.read(tbuf)
        if len(buf)==0:
            break
        c += buf.count(fdl)
    f.seek(-1, 2)
    car = f.read(1)
    if car != fdl:
        c += 1
    f.close()
    return c