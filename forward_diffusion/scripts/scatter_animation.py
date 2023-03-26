from PIL import Image
import glob


frames = []
num_pics = 17
for i in range(num_pics):
    new_frame = Image.open('../images/png/step'+str(i+2)+'.png')
    frames.append(new_frame)

frames[0].save('../png_to_gif.gif', format='GIF',
               append_images=frames,
               save_all=True,
               duration=400, loop=0)