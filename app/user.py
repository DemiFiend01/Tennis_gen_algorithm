import numpy
import pygad
import pygad.nn
import pygad.gann

import pygame 
import gymnasium as gym
import FixedAtariWrapper as faw
import time
from ocatari.core import OCAtari

def map_input_to_action(keys):
    ''' Maps key inputs to Tennis game actions. '''

    # 0. Check if player is shooting

    shooting = keys[pygame.K_SPACE]

    # 0.1 Diagonal shot

    if keys[pygame.K_w] and keys[pygame.K_d] and shooting:
        return 14
    elif keys[pygame.K_w] and keys[pygame.K_a] and shooting:
        return 15
    elif keys[pygame.K_s] and keys[pygame.K_d] and shooting:
        return 16
    elif keys[pygame.K_s] and keys[pygame.K_a] and shooting:
        return 17

    # 0.2 Vertical or horizontal shot

    if keys[pygame.K_w] and shooting:
        return 10
    elif keys[pygame.K_d] and shooting:
        return 11
    if keys[pygame.K_a] and shooting:
        return 12
    elif keys[pygame.K_s] and shooting:
        return 13

    # 1. Diagonal movement

    if keys[pygame.K_w] and keys[pygame.K_d]:
        return 6
    elif keys[pygame.K_w] and keys[pygame.K_a]:
        return 7
    elif keys[pygame.K_s] and keys[pygame.K_d]:
        return 8
    elif keys[pygame.K_s] and keys[pygame.K_a]:
        return 9

    # 2. Vertical & horizontal movement

    if keys[pygame.K_w]:
        return 2
    elif keys[pygame.K_d]:
        return 3
    if keys[pygame.K_a]:
        return 4
    elif keys[pygame.K_s]:
        return 5
    
    # 2. Serving

    if keys[pygame.K_SPACE]:
        return 1
    
    # 3. No action

    return 0

def tennis_playable():
    ''' Tennis game where user can play on the keyboard. '''
    
    # 0. Init pygame window
    
    pygame.init()
    screen = pygame.display.set_mode((100, 100)) # Dummy window for capturing user input
    clock = pygame.time.Clock()

    # 1. Init the environment

    env = OCAtari("Tennis-v4", mode="ram", hud=True, render_mode="human")
    obs, info = env.reset()

    i = 0
    while True:

        # 2. Read pygame events

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Quitting")
                pygame.quit()
                quit()
            
        # 3. Read keyboard input

        keys = pygame.key.get_pressed()
        action = map_input_to_action(keys)

        # 4. Pass action into the game

        obs, reward, terminated, truncated, info = env.step(action)
        #labels = info["labels"]

        i = (i + 1) % 30
        if i == 0:
            for obj in env.objects:
                if obj.category == 'Ball':
                    pass #print(f"Ball: ({obj.x}, {obj.y})")
                elif obj.category == 'Player':
                    print(f"Player: ({obj.x}, {obj.y})")
                elif obj.category == "Enemy":
                    pass#print(f"Enemy: ({obj.x}, {obj.y})")

            RAM = env.get_ram()
            #print(f"Score: ({RAM[69]}, {RAM[70]})")
        
        #print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        # 5. Update pygame clock

        #screen.fill("purple")
        #pygame.display.flip()
        clock.tick(120)