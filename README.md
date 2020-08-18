# Sign-Flip-Attack
This is the **Pytorch** code of our ECCV2020 paper "Boosting Decision-based Black-box Adversarial Attacks with Random Sign Flip" (SFA). In this paper, we proposed a simple and efficient decision-based black-box l-inf adversarial attack.

# Dependencies
* Python 3.6
* Pytorch 1.1.0
* torchvision 0.3.0
* PIL

# Usage
We provide an example of how to perform targeted and untargeted attacks with SFA in `test.py`. `original_img.png` and `target_img.png` are randomly selected from ImageNet. <br>
Run ```CUDA_VISIBLE_DEVICES=[gpu id] python test.py```

# Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:

```
@inproceedings{Chen2020boosting,
    title={Boosting Decision-based Black-box Adversarial Attacks with Random Sign Flip},
    author={Chen, Weilun and Zhang, Zhaoxiang and Hu, Xiaolin and Wu, Baoyuan},
    Booktitle = {Proceedings of the European Conference on Computer Vision},
    year={2020}
}
```
