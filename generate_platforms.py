import random
import settings as config
import pandas as pd

distance_min = min(config.PLATFORM_DISTANCE_GAP)
distance_max = max(config.PLATFORM_DISTANCE_GAP)

#return True with a chance of: P(X=True)=1/x
chance = lambda x: not random.randint(0,x)


random.seed(1)


NUM_LEVELS = 1_000


platforms = list()
for _ in range(NUM_LEVELS):
    if len(platforms) == 0:
        platforms.append({
            'x': config.HALF_XWIN - config.PLATFORM_SIZE[0]//2,
            'y': config.HALF_XWIN + config.YWIN/3,
            'w': config.PLATFORM_SIZE[0],
            'h': config.PLATFORM_SIZE[1],
            'initial_bonus' : False,
            'breakable'     : False,
        })
        continue
    
    offset = random.randint(distance_min, distance_max)    
    platform = {
        'x': random.randint(0,config.XWIN-config.PLATFORM_SIZE[0]),     # int
        'y': platforms[-1].get('y') - offset,                           # int
        'w': config.PLATFORM_SIZE[0],                                   # int
        'h': config.PLATFORM_SIZE[1],                                   # int
        'initial_bonus': chance(config.BONUS_SPAWN_CHANCE),             # bool                                    # boolean
        'breakable': chance(config.BREAKABLE_PLATFORM_CHANCE),          # bool                                              # boolea
    }
    platforms.append(platform)

df_platforms = pd.DataFrame(platforms)
df_platforms.to_csv('platforms.csv', index = False)