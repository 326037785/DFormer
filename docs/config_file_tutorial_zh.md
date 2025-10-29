# DFormer 配置文件填写教程

为了方便你在 DFormer/DFormerv2 框架中快速适配新数据集或自定义实验，本文以 `local_configs/template/DFormer_Large.py` 为例，介绍 `config` 文件（`EasyDict` 对象 `C`）中每个关键字段的含义、填写方式及常见注意事项。

## 1. 配置文件的结构

配置文件通常包含两部分：

1. **数据集与图像预处理设置**：可以直接复用 `local_configs/_base_/datasets` 下现有数据集的配置，或根据模板手动填写。字段位于模板文件前半部分，用于指定数据路径、类别、图像尺寸等。【F:local_configs/template/DFormer_Large.py†L18-L43】【F:local_configs/_base_/datasets/NYUDepthv2.py†L5-L55】
2. **模型、训练与日志设置**：用于指定骨干网络、预训练权重、优化器、训练超参数及日志保存路径等。模板文件中后续所有字段均属于此部分。【F:local_configs/template/DFormer_Large.py†L45-L94】

> **建议流程：** 先在 `local_configs/_base_/datasets` 中复制一个与你数据结构最相近的文件并进行修改，再在对应实验目录（如 `local_configs/NYUDepthv2/`）中创建模型配置文件，导入刚才的数据集配置。

## 2. 数据集字段填写

| 字段 | 说明 | 示例或建议 |
| --- | --- | --- |
| `C.dataset_name` | 数据集名称，将用于日志目录等 | 如 `"NYUDepthv2"`【F:local_configs/_base_/datasets/NYUDepthv2.py†L5-L6】 |
| `C.dataset_path` | 数据集根目录，通常是 `datasets/数据集名`，也可改为绝对路径 | `osp.join(C.root_dir, "NYUDepthv2")`【F:local_configs/template/DFormer_Large.py†L18-L21】 |
| `C.rgb_root_folder` / `C.gt_root_folder` / `C.x_root_folder` | RGB、标注和深度（或额外模态）的子目录 | 模板中默认 `RGB` / `Label` / `Depth`，请按自己数据结构调整 | 
| `C.rgb_format` 等 | 文件后缀，需与实际数据一致 | `.jpg`、`.png` 等【F:local_configs/template/DFormer_Large.py†L21-L28】 |
| `C.gt_transform` | 是否对标签做预处理。若标签 0 表示无效区域通常设为 `True` |【F:local_configs/_base_/datasets/NYUDepthv2.py†L10-L14】 |
| `C.x_is_single_channel` | 额外模态是否为单通道（深度/热成像等） | 深度 `.png` 建议设为 `True`【F:local_configs/template/DFormer_Large.py†L24-L27】 |
| `C.train_source` / `C.eval_source` | 训练/验证列表文件（每行对应一张样本） | 建议使用绝对路径或 `osp.join` 组合 | 
| `C.num_train_imgs` / `C.num_eval_imgs` | 样本数量，用于计算每轮迭代次数 | 可通过 `wc -l train.txt` 获得；若不确定可暂设为 `None` 并在运行时检查 | 
| `C.num_classes` / `C.class_names` | 类别数和类名列表 | 确保顺序与标注 ID 一致；类名仅用于日志显示【F:local_configs/_base_/datasets/NYUDepthv2.py†L19-L55】 |

**常见错误提醒：**
- 列表文件路径不存在或文件中包含空行，会导致数据加载失败。
- `C.num_classes` 与实际标签不一致时会触发 shape mismatch 错误。
- 若标签背景值不是 255，请同步更新 `C.background`。

## 3. 图像预处理配置

- `C.background`：无效像素值，默认 255。【F:local_configs/template/DFormer_Large.py†L33-L40】
- `C.image_height` / `C.image_width`：输入尺寸。若数据集尺寸固定，保持默认即可；若需要多尺度训练，可在数据管道中自定义。
- `C.norm_mean` / `C.norm_std`：RGB 归一化参数，遵循 ImageNet 标准。如使用自定义模态，请替换为对应统计量。

## 4. 模型与预训练权重

- `C.backbone`：选择骨干网络名称（如 `DFormer-Tiny`、`DFormerv2-L`）。需与加载的权重文件保持一致。【F:local_configs/NYUDepthv2/DFormer_Tiny.py†L4-L7】
- `C.pretrained_model`：预训练权重的路径，可是相对路径（例如 `checkpoints/pretrained/DFormer_Tiny.pth.tar`）。确保文件存在，否则会在加载时抛出错误。
- `C.decoder` / `C.decoder_embed_dim`：解码器类型与通道数。默认 `ham` 与 512，若自定义解码器请保持与实现一致。
- `C.optimizer`：目前支持 `AdamW`，如需更换请确认优化器实现已在训练脚本中支持。
- `C.channels`：各阶段通道数配置，与骨干网络结构对应，仅在模板中出现，若使用 `_base_` 配置请确认是否需要保留或修改。【F:local_configs/template/DFormer_Large.py†L59-L63】

## 5. 训练超参数

- `C.lr`、`C.lr_power`、`C.weight_decay` 等控制学习率策略和正则项，可按常规经验调整。【F:local_configs/template/DFormer_Large.py†L45-L58】
- `C.batch_size`：单卡 batch 大小；多卡训练请参考启动脚本中的 `--batch-size` 参数。
- `C.nepochs`：总训练轮数。
- `C.niters_per_epoch`：每轮迭代数，模板中通过 `C.num_train_imgs // C.batch_size + 1` 计算。若 `C.num_train_imgs` 未设置，需要在运行前补充，否则会报 `TypeError`。【F:local_configs/template/DFormer_Large.py†L52-L58】
- `C.num_workers`：`DataLoader` 的并行度，视机器性能调整。
- `C.train_scale_array`、`C.warm_up_epoch`：多尺度训练倍率与 warmup 轮数，可按需修改。
- `C.drop_path_rate`、`C.aux_rate`：结构相关超参。不同模型推荐值可参考现有配置文件。

## 6. 验证与测试设置

- `C.eval_iter`：训练过程中多少个 epoch 触发一次验证。【F:local_configs/template/DFormer_Large.py†L65-L74】
- `C.eval_stride_rate` / `C.eval_scale_array` / `C.eval_flip`：控制滑窗测试、尺度与翻转增强。
- `C.eval_crop_size`：裁剪尺寸，需与模型输入尺寸兼容。
- `C.is_test`：若希望仅生成提交结果，可在数据集配置中根据需要调整。

## 7. 日志与检查点路径

- `C.log_dir`：日志主目录，默认会包含数据集名与骨干名，可根据需要加时间戳避免覆盖。【F:local_configs/NYUDepthv2/DFormer_Tiny.py†L29-L41】
- `C.tb_dir`：TensorBoard 目录。
- `C.checkpoint_dir`：模型权重保存位置，确保目录可写。
- `C.log_file` / `C.val_log_file`：训练与验证日志文件名称。

> **提示：** 模板未自动创建目录，建议在初始化脚本或手动运行 `os.makedirs(config.log_dir, exist_ok=True)` 以避免路径不存在带来的错误。参考 `local_configs/NYUDepthv2/DFormer_Tiny.py` 中的处理方式。【F:local_configs/NYUDepthv2/DFormer_Tiny.py†L33-L41】

## 8. 实践流程范例

1. 复制模板：`cp local_configs/template/DFormer_Large.py local_configs/MyDataset/DFormer_Large.py`。
2. 根据数据实际结构修改“数据集字段”和“图像预处理配置”。
3. 将预训练模型（若有）放入 `checkpoints/pretrained/` 并更新路径。
4. 调整训练与验证超参数。
5. 在训练脚本中指定新的配置文件，例如：
   ```bash
   python debug_main.py \
       --config local_configs/MyDataset/DFormer_Large.py \
       --launcher none
   ```
6. 启动训练后，检查日志目录与 `tensorboard`，确认路径与样本数量均正确。

## 9. 常见排查方法

- **路径问题**：运行前使用 `python -c "from local_configs.MyDataset.DFormer_Large import config; print(config.dataset_path)"` 检查路径是否正确。
- **类别不匹配**：观察训练日志中的 `num_classes` 输出或在数据加载脚本中打印标签统计。
- **额外模态通道数**：若深度图是 3 通道伪彩图，请将 `C.x_is_single_channel` 设为 `False` 并在数据加载时确认维度。

通过以上步骤即可完成自定义配置文件的编写与校验。如果仍有疑问，建议对照 `local_configs/NYUDepthv2/` 与 `local_configs/SUNRGBD/` 中的现有配置进行参考。
