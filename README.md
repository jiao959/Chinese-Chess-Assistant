# 中国象棋辅助分析 MVP

这是一个本地运行的中国象棋辅助分析工具。程序会截取屏幕上的棋盘，识别当前棋局，转换为 Xiangqi FEN，调用本地 Pikafish 引擎，并在窗口中显示己方最佳走法和箭头示意图。

本项目不包含自动落子、自动点击、模拟鼠标走棋功能。

## 这是什么类型的项目

这是一个 Python 桌面 App 项目，不是网页项目，也不是单纯的命令行脚本。

启动方式是运行：

```powershell
python main.py
```

运行后会打开一个 Windows 桌面窗口。用户通过窗口里的按钮完成截图、识别和分析。

项目里也包含少量辅助脚本，例如：

```text
make_templates_from_startpos.py
```

这个脚本只用于从标准开局图片生成棋子模板，不是主程序入口。日常使用主程序时只需要运行 `main.py`。

## 如何使用这个项目

第一次使用建议按这个顺序：

1. 安装 Python 依赖。
2. 把 Pikafish 引擎和 `pikafish.nnue` 放到 `engines` 目录。
3. 准备或更新 `templates` 棋子模板。
4. 运行 `python main.py` 打开桌面窗口。
5. 打开你的中国象棋游戏或模拟器，让棋盘显示在屏幕上。
6. 在本工具里点击 `分析最佳走法`，程序会自动截图、裁剪棋盘、识别棋子、调用引擎，并显示最佳走法。
7. 如果自动全屏捕捉不准，可以点击 `手动选择区域`，框选棋盘所在窗口或区域，程序会在你框选的区域里继续自动裁剪棋盘并分析。

## 打开后的 UI 说明

窗口顶部有两个主要按钮：

- `分析最佳走法`：从全屏开始自动捕捉棋盘，然后完成裁剪、识别、分析和箭头显示。
- `手动选择区域`：让你先用鼠标框选一个大致区域，后续流程和 `分析最佳走法` 一样。

窗口中部是状态和参数：

- `己方颜色显示`：程序自动判断棋盘下方是哪一方。
- `分析方`：显示当前按哪一方计算最佳走法。这里始终是“己方”，不是判断真实轮到谁走。
- `手动选择己方颜色`：如果自动判断错误，可以手动选择红方或黑方。
- `分析模式`：选择按时间分析或按深度分析。
- `思考时间`：按时间模式下 Pikafish 思考多少毫秒。
- `搜索深度`：按深度模式下 Pikafish 搜索到多少层。
- `线程`：Pikafish 使用的 CPU 线程数。
- `Hash`：Pikafish 使用的缓存内存大小。
- `中文走法`：显示最终最佳走法，例如 `马四进五`。

窗口下方是棋盘预览区：

- 分析完成后会显示裁剪后的棋盘。
- 如果引擎返回了有效走法，会在棋盘上画橙色箭头。
- 如果识别错误，通常需要检查 `debug_outputs` 目录里的裁剪图、交叉点图和识别矩阵。

## 功能

- PySide6 桌面窗口。
- 自动捕捉屏幕并裁剪棋盘。
- 支持手动框选一个初始区域，再在该区域内自动裁剪棋盘。
- 使用 OpenCV + 本地模板识别棋子。
- 自动判断棋盘下方己方颜色。
- 调用本地 Pikafish 分析最佳走法。
- 显示中文四字走法，例如 `马四进五`。
- 在棋盘预览图上画出最佳走法箭头。
- 输出 debug 文件，便于检查裁剪、交叉点和识别结果。

## 项目结构

```text
.
├── main.py                         # 程序入口和 PySide6 UI
├── screen_capture.py               # 截屏和手动框选区域
├── board_cropper.py                # 自动裁剪棋盘
├── board_recognizer.py             # 棋盘交叉点定位和棋子模板识别
├── fen_converter.py                # 棋盘矩阵和 FEN 转换
├── engine_client.py                # Pikafish UCI 调用
├── move_notation.py                # bestmove 转中文走法
├── move_overlay.py                 # 最佳走法箭头绘制
├── make_templates_from_startpos.py # 从标准开局图生成模板
├── config.json                     # 配置文件
├── requirements.txt                # Python 依赖
├── templates/                      # 棋子模板
├── engines/                        # Pikafish 引擎和 nnue 文件
├── debug_outputs/                  # 每次运行生成的调试输出
└── tests/                          # 基础测试
```

## 安装依赖

进入项目目录：

```powershell
cd "C:\Users\jh\Desktop\象棋辅助"
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 当前包含：

```text
PySide6
opencv-python
numpy
mss
Pillow
```

## 放置 Pikafish

把 Pikafish 引擎和神经网络文件放到：

```text
C:\Users\jh\Desktop\象棋辅助\engines
```

默认配置要求：

```text
engines\pikafish.exe
engines\pikafish.nnue
```

如果你的 exe 名字不同，例如：

```text
pikafish-avxvnni.exe
```

可以改 `config.json`：

```json
"pikafish_path": "engines/pikafish-avxvnni.exe"
```

如果缺少 `pikafish.nnue`，Pikafish 会退出并提示神经网络文件未加载。

## 准备棋子模板

模板目录是：

```text
templates
```

每种棋子一个文件夹：

```text
red_king
red_advisor
red_bishop
red_rook
red_knight
red_cannon
red_pawn
black_king
black_advisor
black_bishop
black_rook
black_knight
black_cannon
black_pawn
```

可以使用标准开局图自动生成模板：

```powershell
python make_templates_from_startpos.py --image "C:\Users\jh\Desktop\standard.png"
```

运行后重点检查：

```text
templates\_template_source_capture.png
templates\_grid_preview.png
```

`_grid_preview.png` 里的红点应该落在棋盘交叉点中心。如果红点偏了，生成的模板质量会差。

注意：`make_templates_from_startpos.py` 只适合标准开局图。如果图片不是标准开局，它会把错误位置的棋子裁进模板。

## 启动程序

```powershell
python main.py
```

## 使用方式

### 分析最佳走法

点击：

```text
分析最佳走法
```

流程：

```text
全屏截图
-> 自动裁剪棋盘
-> 识别棋子
-> 自动判断己方颜色
-> 生成 FEN
-> 调用 Pikafish
-> 显示中文走法
-> 在预览图上画箭头
```

### 手动选择区域

点击：

```text
手动选择区域
```

用鼠标框选棋盘所在的大致窗口或区域。

流程：

```text
框选区域
-> 在框选区域内自动裁剪棋盘
-> 识别棋子
-> 自动判断己方颜色
-> 生成 FEN
-> 调用 Pikafish
-> 显示中文走法
-> 在预览图上画箭头
```

手动选择区域只是把初始截图范围从“全屏”改成“你框选的区域”，后续流程和“分析最佳走法”一致。

## 引擎参数说明

UI 中有这些参数：

```text
分析模式
思考时间
搜索深度
线程
Hash
```

说明：

- `分析模式 = 按时间`：使用 `go movetime`，思考固定毫秒数。
- `分析模式 = 按深度`：使用 `go depth`，尽量搜索到指定深度。
- `思考时间`：单位毫秒，`1000 ms = 1 秒`。只在按时间模式下主要生效。
- `搜索深度`：只在按深度模式下主要生效。越大越慢。
- `线程`：Pikafish 使用的 CPU 线程数。
- `Hash`：Pikafish 使用的缓存内存，单位 MB。

建议先用：

```text
分析模式：按时间
思考时间：2000 ms
线程：4 或 8
Hash：256 MB 或 1024 MB
```

如果使用按深度模式，深度 `18`、`20` 可能会明显变慢。

## 配置文件

配置文件：

```text
config.json
```

常用配置：

```json
{
  "pikafish_path": "engines/pikafish.exe",
  "pikafish_eval_file": "engines/pikafish.nnue",
  "templates_dir": "templates",
  "analysis_movetime_ms": 2000,
  "analysis_mode": "depth",
  "analysis_depth": 20,
  "engine_threads": 8,
  "engine_hash_mb": 1024,
  "template_match_threshold": 0.38
}
```

`last_board_region` 会保存上次棋盘区域，程序会自动更新。

## Debug 输出

每次识别会清理并重新生成：

```text
debug_outputs\full_screen_capture.png       # 全屏截图
debug_outputs\manual_region_capture.png     # 手动框选原始区域
debug_outputs\auto_crop_preview.png         # 自动裁剪预览
debug_outputs\manual_crop_preview.png       # 手动区域内裁剪预览
debug_outputs\cropped_board.png             # 真正用于识别的棋盘图
debug_outputs\last_recognition_grid_preview.png
debug_outputs\recognized_board.txt
debug_outputs\match_details.txt
debug_outputs\bestmove_chinese.txt
debug_outputs\bestmove_preview.png
```

重点检查：

- `cropped_board.png`：裁剪出来的棋盘是否正确。
- `last_recognition_grid_preview.png`：红点是否落在交叉点中心。
- `recognized_board.txt`：棋子矩阵是否识别正确。
- `match_details.txt`：每个已识别棋子的匹配分数和模板来源。
- `bestmove_preview.png`：带箭头的最佳走法示意图。

## 常见问题

### 识别出来的棋子不对

优先检查：

```text
debug_outputs\last_recognition_grid_preview.png
debug_outputs\recognized_board.txt
debug_outputs\match_details.txt
```

如果红点偏离棋子中心，通常是棋盘裁剪或交叉点定位问题。  
如果红点正确但棋子类型错，通常是模板质量或模板数量问题。

### 马和兵偶尔混淆

当前识别仍然基于模板匹配。马和兵在某些截图、阴影、缩放条件下分数可能非常接近。  
可以补充更多实际对局中裁出来的 `red_knight`、`red_pawn`、`black_knight`、`black_pawn` 模板。

### Pikafish 提示 nnue 缺失

确认文件存在：

```text
engines\pikafish.nnue
```

并确认 `config.json`：

```json
"pikafish_eval_file": "engines/pikafish.nnue"
```

### 分析很慢

如果使用：

```text
分析模式：按深度
搜索深度：20
```

可能会明显变慢。可以改成：

```text
分析模式：按时间
思考时间：2000 ms
```

### 箭头起点或终点看起来不合理

通常是识别矩阵错误导致 FEN 错误。先看：

```text
debug_outputs\recognized_board.txt
```

引擎只会根据程序识别出来的棋盘计算，不知道真实屏幕上是否有漏识别的棋子。

## 运行测试

```powershell
python -m unittest discover -s tests
```

语法检查：

```powershell
python -m py_compile main.py board_recognizer.py engine_client.py fen_converter.py move_notation.py move_overlay.py
```

## 当前限制

- 第一版仍是模板匹配，不是深度学习或 OCR。
- 棋子模板质量直接影响识别效果。
- 游戏画面缩放、阴影、动画、高亮提示会影响识别。
- 不支持自动落子。
- 不保证所有棋盘皮肤、所有模拟器缩放比例都稳定。
- 如果识别矩阵错，Pikafish 给出的最佳走法也会跟着错。
