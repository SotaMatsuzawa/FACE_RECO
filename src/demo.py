# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:17:01 2019

@author: souta
"""

"""
このファイルはRubyとのつなぎ込みのデモとして使えるかもしれないです
とりあえず
本番（eval.py）と同じようにリストを返します


"""


def demo(img_path,ckpt_path):
    rate=[{'label': 3, 'name': 'shikaku', 'rate': 79.3}]
    print(rate)
    return rate

if __name__=='__main__':
    demo("画像ファイルのpath","model.ckptのpath")


    
