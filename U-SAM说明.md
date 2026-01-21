# U-SAM 项目概览与复现指南 ✅

## 1) 项目做什么 💡
- 名称：U-SAM — "Tuning Vision Foundation Models for Rectal Cancer Segmentation from CT Scans"。
- 任务：基于 SAM（Segment Anything Model）与自定义下采样 UNet backbone 的组合，针对直肠癌 CT 切片做医学图像分割（CARE 数据集为主，另支持 WORD 数据集）。
- 输出：训练得到的分割模型检查点（.pth）以及 `mean_dice`、`miou` 等评估指标。

---

## 2) 项目结构（关键文件/目录） 🔧
- `u-sam.py`：主训练/评估脚本（命令行参数、数据加载、模型、训练/评估循环、保存 checkpoint）。
- `backbone.py`：自定义下采样 UNet（提取图像特征并输出给 SAM 模块）。
- `dataset/rectum_dataloader.py`：CARE（rectum）数据 loader，读取 `.npz`、CSV、生成 prompt（boxes/points）并做增强。
- `dataset/word_dataloader.py`：WORD 数据 loader。
- `util/`：辅助函数（分布式、collate、metric 计算、保存等）。
- `weight/`：用于放置预训练 SAM 权重（如 `sam_vit_b_01ec64.pth`）。
- `train_sam.sh`：示例多卡分布式训练命令。
- `Annotation_Example/`：若干示例 `.npz` 文件与可视化示例，可参考数据格式。
- `test_paths.py`：用于检查数据目录结构与文件是否就位的小脚本。

---

## 3) 数据与格式要求（DataV6 / CARE） 📁
- 代码默认期望存在名为 `DataV6` 的数据根目录（默认位置：项目父目录下 `DataV6`，即 `../DataV6`）。

建议的数据组织：
```
DataV6/
  ├─ train/
  │   ├─ train_bbox.csv       # CSV: 每行 [basename, bbox]
  │   └─ train_npz/           # 多个 .npz，每个含键: 'image', 'label'
  └─ test/
      ├─ test_bbox.csv
      └─ test_npz/
```
- `.npz` 文件（see `Annotation_Example/`）：包含 `image`（灰度/归一化图像，脚本中会乘 255 再做 CLAHE），和 `label`（标注掩码，整数类标签）。
- CSV：第 1 列为文件名（不含 .npz 后缀），第 2 列为 bbox（字符串形式的 list，例如 `[x1, y1, x2, y2]`）。
- 对于 `word` 数据集，`u-sam.py` 中的 `args.root` 为硬编码路径，需要按本地位置修改或替换为 `DataV6` 结构。

---

## 4) 环境与依赖（建议） 🧩
- Python: 推荐 3.9.x（README 中建议 `python==3.9.12`）。
- PyTorch: `torch==1.11.0`，`torchvision==0.12.0`（匹配 README）。
- 其他：`numpy==1.21.5`、`matplotlib==3.5.2`、`albumentations`、`scipy`、`pandas` 等。
- 必须：在 `weight/` 下放置 SAM 的预训练权重（默认使用 ViT-B）
  - e.g. `weight/sam_vit_b_01ec64.pth`
  - 可从 Segment Anything 官方链接下载（README 中有直链）。

---

## 5) 如何复现（最小步骤） ▶️
1. 克隆或拷贝代码到本地并进入项目目录（确保 `u-sam.py` 在该目录）。
2. 准备数据（CARE）：按上面的 DataV6 结构组织数据；也可参考 `Annotation_Example/` 的示例文件格式。
3. 下载 SAM 预训练权重放到 `weight/`：
   - 常用：`sam_vit_b_01ec64.pth`
4. 安装依赖（示例）:
   - pip install torch==1.11.0 torchvision==0.12.0 numpy==1.21.5 matplotlib albumentations scipy pandas
5. 检查数据路径（可运行）：
   - `python test_paths.py` （脚本会检测 DataV6 的 train/test、CSV 与 .npz 文件）
6. 单卡训练（CARE）：
   - `python u-sam.py --epochs 100 --batch_size 24 --dataset rectum`
7. 多卡训练（示例，8 卡 DDP）：
   - `bash train_sam.sh` 或参照 `train_sam.sh` 中的命令修改 `CUDA_VISIBLE_DEVICES` 等。
8. 评估：
   - `python u-sam.py --dataset rectum --eval --resume /path/to/checkpoint.pth`
   - 输出在控制台及 `exp/U-SAM-Rectum/prompt=<...>/log.txt` 中记录日志和最佳 checkpoint（例如 `best_<mean_dice>_<miou>.pth`）。

---

## 6) 模型与训练细节（如何做的） ⚙️
- Model: `SAM` wrapper（在 `u-sam.py` 中）结合下采样 `UNet`（`backbone.py`）与 SAM 的 image encoder + mask decoder。训练时：
  - 如果选择 prompt（boxes/points），SAM 的 prompt_encoder 会利用它们来生成 sparse/dense embeddings。
  - backbone 输出低分辨率特征供 SAM image_encoder 使用。
- Loss: 混合交叉熵（CE）与 Dice loss（以 0.6 的权重平衡，详见 `calc_loss`）。
- Metric: class-wise Dice 和 IoU，在 `evaluate()` 中汇总并返回 `mean_dice` 与 `miou`。
- Prompt 模式（CLI 参数 `--prompt_mode`）: 0=无提示, 1=GT boxes, 2=GT points, 3=boxes+points（影响训练与 eval 时使用的 prompts）。
- 图像归一化：在 `main()` 中硬编码了 `pixel_mean` 与 `pixel_std`，并在 `SAM.forward` 中进行标准化（所以不要再在外部重复 normalize）。

---

## 7) 常见注意事项 & 排错 🔍
- 确保 `weight/` 下有对应的 SAM 预训练权重，默认是 `vit_b`。
- 数据格式错误（CSV 或 .npz 错误）会导致 dataloader 报错；先用 `test_paths.py` 验证。
- 若使用 `--dataset word`，需修改或传入合适的 `args.root`，否则默认路径为空或为硬编码路径。
- 当训练出现显存不足，可尝试减小 `--batch_size` 或用梯度累积/更小图像尺寸（`--img_size`）。
- 训练过程中输出日志会写入 `exp/U-SAM-Rectum/prompt=<...>/log.txt`，训练检查点同目录保存。

---

## 8) 小贴士 ✅
- 查看 `Annotation_Example/` 来了解 `.npz` 文件和标签组织。
- 若要复现实验论文中的具体超参（如训练轮数、batch size），参见 `README.md` 中推荐设置。
- 想调试小样本或可视化单张结果：在 `u-sam.py` 中开启 `--eval` 并传入 `--resume`，`evaluate(..., visual=True)` 会打印并可视化样例（代码中已有 `visualize()` 调用点，需确保可视化函数存在/启用）。

---

如果你希望，我可以：
1) 根据你本地的数据位置，生成一个 `DataV6` 目录结构检查脚本（或更新 `test_paths.py`），或者
2) 帮你一步步执行从创建虚拟环境、安装依赖到运行一次小规模训练/评估的具体命令。🔧

---

## 9) 使用训练好的 `.pth` 处理 `.npz` 文件（推理） 🔍

下面给出两种常见做法：批量评估（老脚本）与单张/自定义 `.npz` 推理并保存预测（推荐）。

### 方法 A — 使用 `u-sam.py` 的 `--eval`（适合已按 DataV6 组织的测试集）
- 准备 `DataV6/test/test_npz` 与 `test_bbox.csv`（确保文件名在 CSV 中能找到）。
- 运行命令：

```bash
python u-sam.py --dataset rectum --eval --resume /path/to/best_<mean_dice>_<miou>.pth
```

- 输出：控制台显示评估指标（mean_dice, miou）；脚本默认不将单张预测保存到磁盘，如需保存参见方法 B。

### 方法 B — 推荐：单张 `.npz` 推理并保存结果（示例脚本）
- 使用场景：调试、可视化或对任意 `.npz` 逐个推理并把预测 mask 保存为 `.npz`/`.png`。
- 新建 `inference_npz.py`（放在项目根目录），以下为最小示例：

```python
import argparse
import numpy as np
import torch
from util.misc import nested_tensor_from_tensor_list
from u-sam import parse_args, SAM

# 简化示例，不含全部异常处理

def load_npz_and_preprocess(path, img_size):
    npz = np.load(path)
    img = npz['image']  # assumed grayscale normalized [0,1]
    # apply same CLAHE preprocessing as dataloader if desired (or reuse albumentations)
    # resize to img_size
    from scipy.ndimage import zoom
    h, w = img.shape
    if (w, h) != (img_size, img_size):
        img = zoom(img, (img_size / w, img_size / h), order=3)
    img = np.uint8(img * 255)
    # convert to 3-channel float tensor
    img = img.astype(float) / 255
    img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1).float()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    # build args for model like in u-sam
    am = argparse.Namespace()
    am.img_size = args.img_size
    am.sam_num_classes = 3
    am.use_gt_box = False
    am.use_gt_pts = False
    am.use_psd_box = False
    am.use_psd_pts = False
    am.use_psd_mask = False
    am.sam_weight = 'weight/sam_vit_b_01ec64.pth'

    model = SAM(am)
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    img = load_npz_and_preprocess(args.npz, args.img_size)
    samples = nested_tensor_from_tensor_list([img])  # wraps tensor as NestedTensor

    # provide a dummy target (model's eval branch expects targets list)
    dummy_mask = torch.zeros((args.img_size, args.img_size), dtype=torch.long)
    target = {'mask': dummy_mask, 'id': torch.tensor(0)}

    with torch.no_grad():
        outputs = model(samples.to(device), [target])
    # model in eval mode returns (masks, dice_a, dice_b, iou_a, iou_b)
    pred_masks = outputs[0]
    pred = pred_masks[0].cpu().numpy().astype(np.uint8)

    # save prediction
    out_path = args.npz.replace('.npz', '_pred.npz')
    np.savez_compressed(out_path, mask=pred)
    print('Saved prediction to', out_path)

if __name__ == '__main__':
    main()
```

- 使用示例：

```bash
python inference_npz.py --npz DataV6/test/test_npz/0001.npz --resume exp/U-SAM-Rectum/prompt=no_prompt/best_0.656473_0.491037.pth --img_size 224 --device cuda
```

- 可选：把保存的 `.npz` 文件中的 `mask` 用 `matplotlib` 或其他工具保存为 `.png` 以便查看。

### 小提示
- 若你希望批量对一个文件夹运行该脚本，可把上面的步骤放入循环并把结果存到指定输出目录。
- 如需高质量可视化或后处理（连通域、平滑等），建议在保存后再行处理。

---

作者：自动生成总结（基于仓库中的 `README` 与源码解析）
