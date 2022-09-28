## **Implementation**
#### ** Projected GANS Converge Faster Replica**

If you find our code or paper useful, please cite
```bibtex
@InProceedings{Sauer2021NEURIPS,
  author         = {Axel Sauer and Kashyap Chitta and Jens M{\"{u}}ller and Andreas Geiger},
  title          = {Projected GANs Converge Faster},
  booktitle      = {Advances in Neural Information Processing Systems (NeurIPS)},
  year           = {2021},
}
```
<ul>

<li>I have edited and added further lines to the code base.</li>
<li>The <strong>styleGAN</strong> model has a bug and requires further configuration; however, both <strong>FastGAN</strong> and <strong>FastGAN_Lite</strong> work.</li>
<li>Training requires high GPU-RAM and doesn't even train in a Google-Colab Pro environment.</li>
<li>I have used <i># added</i> comment to the lines I added.</li>
<li>Running in more than one GPU causes an error and requires further debugging and configuration.</li>
<li><a href="https://colab.research.google.com/drive/1sWZfNj24RLciYa00323FFO8l2PrD8k_W?usp=sharing">Link to .ipynb file</a></li>
<li><a href="https://drive.google.com/drive/folders/1Wwb-6fYBHo5K6qEdToeCPr1x04pjVZiJ?usp=sharing">Link to entire folder</a></li>


<li>To run script</li>

```
python train.py --outdir=./training-runs/ --cfg=fastgan --data=..\pokemon --gpus=1 --batch=64 --alpha=100

```

```--alpha``` specifies the input to the 4th channel when creating RGBA image. Default value = 255

Upon execution ```pokemonrgba``` folder is created containing pictures from the pokemon dataset.
The pictures have a transparency value equal to alpha
</ul>


## **Files**

1. `rgb2rgba.py` located in `projected_gan` folder
2. `pokemon.zip` contains RGB pictures
3. `pokemonrgba.zip` contains RGBA pictures - only few images are generated for now due to high memory requirement.
3. `projected_gan.zip` contains edited code files


