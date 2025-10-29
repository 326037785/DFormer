# Windows Dataset and Checkpoint Layout

This project expects RGB-D datasets and checkpoints to follow the same folder
layout that is documented in the main [README](README.md). To make it easier to
set up the environment on Windows, the steps below show how to prepare the
`datasets` and `checkpoints` directories.

## 1. Create the directories

Open **PowerShell** in the repository root and run:

```powershell
New-Item -ItemType Directory -Force -Path datasets
New-Item -ItemType Directory -Force -Path checkpoints\pretrained
New-Item -ItemType Directory -Force -Path checkpoints\trained
```

If you keep the data on another drive, you can create a directory junction that
points back to the project so the training scripts still find the files:

```powershell
New-Item -ItemType Directory -Force -Path D:\RGBDData\datasets
New-Item -ItemType SymbolicLink -Path .\datasets -Target D:\RGBDData\datasets
New-Item -ItemType Directory -Force -Path D:\RGBDData\checkpoints\pretrained
New-Item -ItemType Directory -Force -Path D:\RGBDData\checkpoints\trained
New-Item -ItemType SymbolicLink -Path .\checkpoints -Target D:\RGBDData\checkpoints
```

> 💡  `mklink /J` offers the same result from **Command Prompt** if you prefer
> the classic terminal.

## 2. Download datasets

Download the datasets from the links in the main README and arrange the files as
follows:

```
datasets
├── DatasetName
│   ├── RGB
│   │   ├── <image1>.<ext>
│   │   └── <imageN>.<ext>
│   ├── Depth
│   │   ├── <depth1>.<ext>
│   │   └── <depthN>.<ext>
│   ├── train.txt
│   └── test.txt
└── ...
```

Each dataset uses `RGB` and `Depth` subfolders plus the text files that define
the splits. Matching filenames across the two subfolders is important so that
the dataloader can pair color and depth frames correctly.

## 3. Place pre-trained weights

Download the pre-trained weights or fine-tuned checkpoints listed in the README
and copy them into `checkpoints\pretrained` (for ImageNet or released
checkpoints) or `checkpoints\trained\<DatasetName>` (for your own training
runs). Example layout:

```
checkpoints
├── pretrained
│   ├── DFormer_Large.pth.tar
│   ├── DFormer_Base.pth.tar
│   ├── DFormer_Small.pth.tar
│   ├── DFormer_Tiny.pth.tar
│   ├── DFormerv2_Large_pretrained.pth
│   ├── DFormerv2_Base_pretrained.pth
│   └── DFormerv2_Small_pretrained.pth
└── trained
    ├── NYUDepthv2
    │   └── <experiment checkpoints>
    └── SUNRGBD
        └── <experiment checkpoints>
```

The local configuration files already point to these locations, so keeping the
same folder names avoids additional changes when you run `train.bat`,
`eval.bat`, or `infer.bat`.

## 4. Verify the setup

After the files are in place, you can quickly check that everything is
accessible by listing the folders:

```powershell
Get-ChildItem -Recurse -Depth 2 .\datasets
Get-ChildItem -Recurse -Depth 2 .\checkpoints
```

You are now ready to train or evaluate the models on Windows.
