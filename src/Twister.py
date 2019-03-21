#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Input import Input
from Scene import Scene
import sys
import getopt
import Constants
import pygame

class Twister():

    def __init__(self):
        self.input = Input()
        pygame.init()
        pygame.display.set_mode((Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT))
        pygame.display.set_caption("Twister!")
        screen = pygame.display.get_surface()
        self.scene = Scene(screen, self.input)

    def run(self):
        while True:
            self.input.run()
            self.scene.run()



if __name__ == "__main__":
    options, remainder = getopt.getopt(sys.argv[1:], 's:x:')
    for opt, arg in options:
        if opt in ('-s'):
            song = arg
        elif opt in ('-x'):
            speed = float(arg)
    game = Twister()
    game.run()
