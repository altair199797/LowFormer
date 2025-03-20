import torch, os, sys, math
from tqdm import tqdm
from termcolor import colored
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor or int or float) -> torch.Tensor or int or float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def imagenet_eval(model):
    # ImageNet path
    imagenet_path = os.path.join("..","datasets","val")
    image_size = 224
    batch_size = 50
   
   

    model.eval()
    model(torch.randn(1,3,224,224)) # sanity test
    model.cuda()

    # Data loading
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            imagenet_path,
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(image_size / 0.95)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)

    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc="Eval "+ colored(model.name,"light_green")+" on ImageNet") as t: # light_grey
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()


                # compute output
                output = model(images)

                bsize = images.shape[0]
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)
    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")


def lowformer_imagenet_eval(modelname):
    from lowformer_model import get_lowformer, get_model_by_name
    model = get_model_by_name(modelname)
    imagenet_eval(model)


if __name__ == "__main__":
    lowformer_imagenet_eval("b3")