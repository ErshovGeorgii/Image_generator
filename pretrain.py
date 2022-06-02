from this import d
from PIL import Image, ImageDraw
import random
import statistics
import math
import numpy as np
import pandas as pd
import os

def norm_pdf(x, mu, sigma):
    f_x = math.exp(-1/2 *  ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))
    return f_x

size_of_pic = 1024

# Генерация картинок
def img_generate(noize, sigma):
    img = Image.new('L', (size_of_pic, size_of_pic), 0)
    wigth = img.size[0]
    heigth = img.size[1]
    pix = img.load()
    draw = ImageDraw.Draw(img)
    stars = []
    circles = []

    n = random.randint(10, 60)
    p = random.randint(10, 15)

    max_star_size = 6.5
    max_dist = 50
    

    # Adding noise
    for i in range(wigth):
        for j in range(heigth):
            noizetion = pix[i, j] + random.randint(0, int(noize * 255)) 
            draw.point((i, j), noizetion)

    # Adding hot pixels
    for i in range(100):
        x, y = random.randint(0, size_of_pic), random.randint(0, size_of_pic)
        hot_pixel_level = random.randint(1, 255)
        draw.point((i, j), hot_pixel_level)

    # Adding stars coordinates
    for i in range(n):
        x, y = random.randint(0, size_of_pic), random.randint(0, size_of_pic)
        star_size = max_star_size - 3 + 3 * random.uniform(0, 1)
        stars.append([x, y, star_size, 1])
    
    # Drawing stars 
    for k in range(n):
        inten = min(stars, key = lambda x: x[2])[2]/stars[k][2]
        sigma_star = (max_star_size - stars[k][2]) * sigma

        x, y = stars[k][0], stars[k][1]

        for i in range(-max_dist, max_dist):
            for j in range(-max_dist, max_dist):
                if x + i > 1 and x + i < size_of_pic and y + j > 1 and y + j < size_of_pic:
                    dist = math.sqrt(i*i + j*j)
                    pix_light = pix[x + i,y + j] + int(255 * inten * norm_pdf(dist, 0, sigma_star) / norm_pdf(0, 0, sigma_star))
                    if pix_light < pix[x + i, y + j]:
                        pix_light = pix[x + i, y + j]
                    elif pix_light > 255:
                        pix_light = 255
                    draw.point((x + i, y + j), pix_light)
    
    # Adding circles coordinates
    for l in range(p):
        x, y = random.randint(0, size_of_pic), random.randint(0, size_of_pic)
        circle_size = max_star_size - 3 + 3 * random.uniform(0, 1)
        circles.append([x, y, circle_size, 2])

    for l in range(p):
        inten = min(circles, key = lambda x: x[2])[2]/circles[l][2]
        sigma_circle = (max_star_size - circles[l][2]) * sigma
        x, y = circles[l][0], circles[l][1]
        for i in range(-max_dist, max_dist):
            for j in range(-max_dist, max_dist):
                if x + i > 1 and x + i < size_of_pic and y + j > 1 and y + j < size_of_pic:
                    dist = math.sqrt(i*i + j*j)
                    if dist < sigma_circle:
                        pix_light = pix[x + i, y + j] + int(inten * 255)
                        if pix_light < pix[x + i, y + j]:
                            pix_light = pix[x + i, y + j]
                        elif pix_light > 255:
                            pix_light = 255
                    else:
                        pix_light = pix[x + i, y + j]
                    draw.point((x + i, y + j), pix_light)
    

    #img.show()
    image = list(img.getdata())
    return (img, [image, stars + circles])



# Generating images
n = 20
noise_step = 0.05
sigma_step = 0.5
true_path = "/home/alex/Desktop/Gosha/my_data/all_data"
image_path = "/home/alex/Desktop/Gosha/my_data/images"
data_train = []
#Generating all images in one click
for s_i in range(1, 7):
    sigma = round(sigma_step * s_i, 2)
    for n_i in range(1, 11):
        noise = round(noise_step * n_i, 2)
        path = "{}/sigma_{}/noise_{}".format(image_path, sigma, noise)
        try:
            os.makedirs(path)
        except: 
            pass


        for i in range(n):
            one_image = img_generate(noise, sigma)
            one_image[0].save("{}/sigma_{}/noise_{}/image_{}_{}_{}.jpg".format(image_path, sigma, noise, sigma, noise, i))
            one_image[0].save("{}/image_{}_{}_{}.jpg".format(true_path, sigma, noise, i))
            data_train.append(["image_{}_{}_{}".format(sigma, noise, i), one_image[1][1]])

        #pd.DataFrame(data_train).to_csv("{}/data_sigma_{}_noise_{}.csv".format(path, sigma, noise))
        
df = pd.DataFrame(data_train)
df.to_csv("{}/data.csv".format(true_path))