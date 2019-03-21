# -*- coding: utf-8 -*-
from Input import Input
import pygame
import Constants


class Scene():
    def __init__(self, screen, input):
        self.input = input
        self.screen = screen
        
        self.sceneClock = pygame.time.Clock()
        self.backgroundColor = (0,0,0)

    def renderWebCam(self):
        frame = self.input.getCurrentFrameAsImage()
        self.screen.blit(frame, (0,0))

    def render(self):
        self.renderWebCam()

    def run(self):
        self.screenDelay = self.sceneClock.tick()
        self.screen.fill(self.backgroundColor)
        self.render()
        pygame.display.flip()
