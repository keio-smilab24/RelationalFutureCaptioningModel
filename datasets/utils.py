from typing import Dict, Any

from torchvision import transforms



def get_params(
    dataset_name: str,
) -> Dict[str, Any]:
    params = dict()

    params["name"] = dataset_name
    
    # 画像関連の平均と分散
    # ロボット視点のrgb画像
    params["img_rgb_mean"] = (0.5, 0.5, 0.5)
    params["img_rgb_std"] = (0.5, 0.5, 0.5)

    #　ロボット視点のdepth画像
    params['img_depth_mean'] = (0.5, 0.5, 0.5)
    params['img_depth_std'] = (0.5, 0.5, 0.5)

    # targetのrgb画像
    params['target_rgb_mean'] = (0.5, 0.5, 0.5)
    params['target_rgb_std'] = (0.5, 0.5, 0.5)

    # targetのdepth画像
    params['target_depth_mean'] = (0.5, 0.5, 0.5)
    params['target_depth_std'] = (0.5, 0.5, 0.5)

    # attention mapのrgb画像
    params['attention_rgb_mean'] = (0.5, 0.5, 0.5)
    params['attnetion_rgb_std'] = (0.5, 0.5, 0.5)

    # attention mapのdepth画像
    params['attention_depth_mean'] = (0.5, 0.5, 0.5)
    params['attention_depth_std'] = (0.5, 0.5, 0.5)

    return params


def create_transform_to_dataset(
    image_size: int,
    img_kind: str,
    dataset_name: str,
    params: Dict[str, Any],
    is_transform: bool = False,
):
    if is_transform:
        pass
    
    return transforms.Compose(
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(params[img_kind], params[img_kind]),
    )