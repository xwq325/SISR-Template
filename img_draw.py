import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image, ImageDraw, ImageFont


def join(image_dir, model_name):
    # 获取文件夹中所有图像文件的列表
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    # 加载每个图像
    images = []
    index = 1

    for image_file in image_files:
        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font="arial.ttf", size=60)
        x = 30
        y = 30
        fill = (0, 255, 0)
        draw.text((x, y), "({}) {}".format(index, model_name[index - 1]), font=font, fill=fill)
        index = index + 1
        images.append(image)

    width = images[0].width
    height = images[0].height
    # 创建一个新的白色背景图像
    new_image = Image.new('RGB', (4 * width + 15, 2 * height + 5), 'white')

    # 将四个图像粘贴到新图像的正确位置
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (5 + width, 0))
    new_image.paste(images[2], (10 + 2 * width, 0))
    new_image.paste(images[3], (15 + 3 * width, 0))
    new_image.paste(images[4], (0, 5 + height))
    new_image.paste(images[5], (5 + width, 5 + height))
    new_image.paste(images[6], (10 + 2 * width, 5 + height))
    new_image.paste(images[7], (15 + 3 * width, 5 + height))

    # 将最终图像保存到磁盘上
    new_image.save(image_dir + 'output.png')


def enlarge_drawing(x, y, image_name, savepath, image, right=False, save=False, scale=0.6):

    np_image = np.array(image)
    if np_image.ndim == 3:
        img_h, img_w, _ = np_image.shape
        width, hight = int(img_w / 10), int(img_h / 10)
        # 先行再列，所以先高再宽
        np_image_part = np_image[y:y + hight, x:x + width, :]
        img_part = Image.fromarray(np_image_part)
    else:
        img_h, img_w = np_image.shape
        width, hight = int(img_w / 3), int(img_w / 3)

        np_image_part = np_image[y:y + hight, x:x + width]
        img_part = Image.fromarray(np_image_part)

    fig = plt.figure(figsize=(10, 10 * img_h / img_w))

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image)

    ax.add_patch(plt.Rectangle((x, y), width, hight, color="red", fill=False, linewidth=3))

    if not right:
        axins = ax.inset_axes((0, int(scale * img_h), int((1 - scale) * img_w), int((1 - scale) * img_h)),
                              transform=ax.transData)
        axins.imshow(img_part)

        ax.add_patch(
            plt.Rectangle((int(0 * img_w), int(scale * img_h)), int((1 - scale) * img_w), int((1 - scale) * img_h),
                          color="blue", fill=False, linewidth=5))

        axins.axis('off')
    else:
        axins = ax.inset_axes(
            (int(scale * img_w), int(scale * img_h), int((1 - scale) * img_w), int((1 - scale) * img_h)),
            transform=ax.transData)
        axins.imshow(img_part)

        ax.add_patch(
            plt.Rectangle((int(scale * img_w), int(scale * img_h)), int((1 - scale) * img_w), int((1 - scale) * img_h),
                          color="blue", fill=False, linewidth=5))

        axins.axis('off')

    save_path = os.path.join(savepath, 'x_{}_y_{}_'.format(x, y)) + image_name[:-3] + "png"
    plt.savefig(save_path)

    if save:
        save_path1 = os.path.join(savepath, 'x_{}_y_{}_part'.format(x, y)) + image_name[:-3] + "png"
        img_part.save(save_path1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Draw')

    parser.add_argument('--input_path', type=str,
                        help='SR images input_path')
    parser.add_argument('--save_path', type=str,
                        help='SR images output_path')
    parser.add_argument('--right', action='store_true',
                        help='put enlarged partial right')
    parser.add_argument('--x', type=int,
                        help='enlarged partial top left corner x')
    parser.add_argument('--y', type=int,
                        help='enlarged partial top left corner y')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='enlarged partial scale')
    parser.add_argument('--model_name', type=str,
                        help='model name with - split')
    parser.add_argument('--save_partial', action='store_true',
                        help='save partial images')
    args = parser.parse_args()
    args.model_name = args.model_name.split('-')

    input_path = args.input_path
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(input_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(input_path, filename)).convert('RGB')
            enlarge_drawing(args.x, args.y, image_name=filename, savepath=save_path, image=image, right=args.right, save=args.save_partial)

    join(save_path, args.model_name)
