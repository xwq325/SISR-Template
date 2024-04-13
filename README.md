# SISR Template

This is my SISR template base on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).

## Add Multiple Scale

Such as:

```
python main.py --scale 2+3+4
```

## Add SSIM Calculation

Base on [EDSR-ssim](https://github.com/HolmesShuan/EDSR-ssim) and [DSRNet](https://github.com/hellloxiaotian/DSRNet/blob/main/DSRNet/sample.py#L56).

Only show the current epoch's avg_SSIM, and provide two implementation method.

If you need the best SSIM, please change the test method in trainer.py

Such as:

![image-20240412044050002](https://cdn.jsdelivr.net/gh/xwq325/PicGo@main/image-20240412044050002.png)

## Add GPU Choice

Such as:

```
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
This is on main.py line 2

Run on GPUs by:
Python main.py --n_GPUs 2
```

## Add Manga109

Such as:

```
python main.py --data_test Manga109
```

## Add Params and Flops

TODO

## Add IDN Implementation

[IDN](arxiv.org/abs/1803.09454): Fast and Accurate Single Image Super-Resolution via Information Distillation Network

## CUDA out of memory

When codes val or test on DIV2K, the error will be happen.

Try to use '--chop' to finish the test.

## Dataset Download

[Benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar): Set5, Set14, B100, U100 with x2, x3, x4.

[Benchmark and DIV2K](https://drive.google.com/drive/folders/1-99XFJs_fvQ2wFdxXrnJFcRRyPJYKN0K): Set5, Set14, B100, U100, Manga109, DIV2K with x2, x3, x4, x8.

But I don't test my template with scale x8 yet.

[DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar): DIV2K.

## Dataset Location

Put datasets like this:

![image-20240412045000065](https://cdn.jsdelivr.net/gh/xwq325/PicGo@main/image-20240412045000065.png)

And change the option.py :

![image-20240412045948359](https://cdn.jsdelivr.net/gh/xwq325/PicGo@main/image-20240412045948359.png)

Or use command line:

```
python main.py --dir_data /MyModel/dataset
```

## Command Line(Linux)

Run as:

```
python main.py --model model_name --data_test DIV2K --data_range 1-800/801-810 --scale 2+3+4 --patch_size 32 --batch_size 16 --save save_dir --n_GPUs 2 --reset
```

Resume as:

```
python main.py --model model_name --data_test DIV2K --data_range 1-800/801-810 --scale 2+3+4 --patch_size 32 --batch_size 16 --load save_dir --n_GPUs 1 --resume -1
```

Resume from N epochs to N+200 epochs as

```
python main.py --model model_name --data_test DIV2K --data_range 1-800/801-810 --scale 2+3+4 --patch_size 32 --batch_size 16 --load save_dir --n_GPUs 1 --resume -1 --epoch N+200
```

--------------------------------------------------------------------------------------------------------------

<a id="BUG1">*BUG:*</a>

*If we resume from the previous epoch, the lr will wrong.*

*Example: When the code resume from 300 epoch, and the previous lr is 1e-4, the previous decay is '200-400-600', the 301 epoch lr will be 2.5e-5 (It should be 5e-5).*

*We need set the decay to '400-600', bacause 200 < 300 < 400.*

*Refer to the [issues](https://github.com/sanghyun-son/EDSR-PyTorch/issues/296)*

--------------------------------------------------------------------------------------------------------------

Test with results picture as:

```
python main.py --model model_name --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2+3+4 --pre_train /xx/xx/model_best.pt --test_only --save_results --reset
```

Test without results as:

```
python main.py --model model_name --data_test Set5+Set14+Urban100+B100 --scale 2+3+4 --pre_train /xx/xx/model_best.pt --test_only --reset
```

## Result Location

Tips: val_dataset is the lastest dataset in "--data_test".

Such as:

```
python main.py --model model_name --data_test B100+DIV2K --data_range 1-800/801-810 --scale 2+3+4 --patch_size 32 --batch_size 16 --save save_dir --n_GPUs 2 --reset
```

Above command line, the val_dataset is DIV2K(801-810), the test_dataset is B100(It doesn't participate in the best model selection).

Such as:

![image-20240412045742460](https://cdn.jsdelivr.net/gh/xwq325/PicGo@main/image-20240412045742460.png)

![image-20240412045821657](https://cdn.jsdelivr.net/gh/xwq325/PicGo@main/image-20240412045821657.png)

## Requirement.txt

You can create a requirement.txt and run by:

```
pip install -r requirement.txt
```

And install by Mirror:

```
pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## Screen or Tmux

Use screen or tmux to run your model. It's convenient.

Create:

```
screen -S name    
tmux new -s name
```

Detach:

```
Ctrl + A + D
Ctrl + B + D
```

Attach:

```
screen -r name   
tmux a -t name
```

Kill:

```
screen -X -S name quit
tmux kill-session -t name // onlt one session
tmux kill-server //all sessions
```

## Environment

I have run it in these environments.

- Python 3.6-3.8
- PyTorch 1.1, 1.2, 1.10, 1.11, 1.12, 1.13
- CUDA 9.2, 10.1, 11.3, 11.6, 11.7
- RTX 2080Ti, RTX 3090 x1, RTX 3090 x2, RTX 3090 x4

Please keep CUDA/PyTorch/Device consistent as far as possible.

## Debug Log

- Plot psnr : Plot four curve on the graph
- config.txt write : Call checkpoint.init circularly when codes run on Windows PyCharm, this lead to write config.txt problem.
- get_lr(): In CustomOptimizer.get_lr(), return lr_scheduler.get_lr(). It should be get_last_lr() in PyTorch >= 1.1.
- resume: Refer to <a href="#BUG1">the Bug. </a>
- chop：When chop is True, many problem will happen. I fix it refer to the [issues](https://github.com/sanghyun-son/EDSR-PyTorch/issues/184).
- And more......

