from torchvision import transforms

#@staticmethod
def transform_simple(size=32):
    transform = transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor(),
                               ])
    return transform

#@staticmethod
def transform_to3ch(size=32):
    transform = transforms.Compose([transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    return transform
