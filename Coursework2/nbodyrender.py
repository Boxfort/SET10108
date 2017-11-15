# Jack Anderson 2017
# Takes in dataset generated from N-body simulation and renders it to an .mp4 video.

from PIL import Image, ImageDraw
import numpy, cv2
import csv
import math

FILENAME = "Coursework2/data.csv"
IMAGE_PREFIX = "Images/IMG"
IMAGE_TYPE = "PNG"
IMAGE_DIM = (1920, 1080)
MODE = "RGB"
ELLIPSE_SIZE = 0.5
VEL_MUL = 2.0
N = 100
IMAGES = 1000
PI = 3.14159265358979323846

def clamp(value, lo, hi):
    if value > hi:
        value = hi
    elif value < lo:
        value = lo
    return value

if __name__ == '__main__':

    image = Image.new(MODE, IMAGE_DIM)
    draw = ImageDraw.Draw(image)

    video = cv2.VideoWriter("nbody.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 100, IMAGE_DIM)

    count = 0
    image_num = 0

    with open(FILENAME, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if float(row[5]) != 0:
                size = math.sqrt(float(row[5]) / PI) * ELLIPSE_SIZE #float(row[5]) * ELLIPSE_SIZE #
                x1 = ((float(row[1]) * (IMAGE_DIM[0])/2) + IMAGE_DIM[0] / 2) - (size/2)
                y1 = ((float(row[2]) * (IMAGE_DIM[1])/2) + IMAGE_DIM[1] / 2) - (size/2)
                x2 = ((float(row[1]) * (IMAGE_DIM[0])/2) + IMAGE_DIM[0] / 2) + (size/2)
                y2 = ((float(row[2]) * (IMAGE_DIM[1])/2) + IMAGE_DIM[1] / 2) + (size/2)
                draw.ellipse((x1, y1, x2, y2), fill=(255, 255 - int(clamp(abs((float(row[3]) + float(row[4]))/2) * 100000, 0, 255)), 255 - int(clamp((abs(float(row[3]) + float(row[4]))/2) * 100000, 0, 255))))

            #x1 = (float(row[1]) * IMAGE_DIM[0]/2) + (IMAGE_DIM[0] / 2)
            #y1 = (float(row[2]) * IMAGE_DIM[1]/2) + (IMAGE_DIM[1] / 2)
            #x2 = (((float(row[1]) - float(row[3])) * IMAGE_DIM[0]/2) * VEL_MUL) + (IMAGE_DIM[0] / 2)
            #y2 = (((float(row[2]) - float(row[4])) * IMAGE_DIM[1]/2) * VEL_MUL) + (IMAGE_DIM[1] / 2)
            #draw.line((x1, y1, x2, y2), fill=128, width=1)
            count += 1
            if count == (N):
                count = 0
                #image.save(IMAGE_PREFIX + str(image_num).zfill(4) + "." + IMAGE_TYPE)
                image_num += 1
                video.write(cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR))
                image = Image.new(MODE, IMAGE_DIM)
                draw = ImageDraw.Draw(image)
                print("Drawing frame " + str(image_num) + " of " + str(IMAGES) + "\r", end='', flush=True)
            #if image_num == 500:
            #    break
        del draw

    video.release()

    print("Done!")
