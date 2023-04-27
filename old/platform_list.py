from new_game import Platform
import pandas as pd

NUM_PLATFORMS = 10_000

platforms = list()

for _ in range(NUM_PLATFORMS):
    platform = Platform()
    
    platforms.append({
        platform.__dict__
    })
    
df_platforms = pd.DataFrame(platforms)
df_platforms.to_csv('platforms.csv',index=False)