import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from SFA import *

if __name__ == '__main__':
    # load images
    def img2tensor(filename):
        img = Image.open(filename)
        transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), ])
        return transform(img).unsqueeze(0)

    original_img = img2tensor('./original_img.png')
    target_img = img2tensor('./target_img.png')

    # initialize target model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet50(pretrained=True)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net.eval()

    original_label = get_predict_label(original_img, net)
    target_label = get_predict_label(target_img, net)
    print(f"original label: {original_label.item()} target label: {target_label.item()}")

    print('##############################')
    print('Untargeted attack')
    adv_img, q = SFA(x=original_img,
                     y=original_label,
                     model=net,
                     resize_factor=2.,
                     x_a=None,
                     targeted=False,
                     max_queries=20000,
                     linf=0.031)
    print(f"original label: {original_label.item()}, "
          f"adversarial label: {get_predict_label(adv_img, net).item()}, "
          f"linf dist: {(adv_img - original_img).abs().max()}, "
          f"queries: {q.item()}")

    print('#############################')

    print('targeted attack')
    adv_img, q = SFA(x=original_img,
                     y=target_label,
                     model=net,
                     resize_factor=2.,
                     x_a=target_img,
                     targeted=True,
                     max_queries=100000,
                     linf=0.031)
    print(f"original label: {original_label.item()}, "
          f"target label: {target_label.item()}, "
          f"adversarial label: {get_predict_label(adv_img, net).item()}, "
          f"linf dist: {(adv_img - original_img).abs().max()}, "
          f"queries: {q.item()}")
