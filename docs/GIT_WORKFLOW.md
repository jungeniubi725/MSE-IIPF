# 使用 Git Bash 管理项目

本项目使用 `main` 分支，GitHub 远程名称为 `origin`。在项目文件夹中打开 Git Bash 管理源码。

仓库主页：[jungeniubi725/MSE-IIPF](https://github.com/jungeniubi725/MSE-IIPF)。源存储库 URL：

```text
https://github.com/jungeniubi725/MSE-IIPF.git
```

检查本地仓库配置：

```bash
git remote get-url origin
git status
```

第一条命令显示可复制的源存储库 URL；第二条显示当前分支、未提交修改和同步状态。私有仓库的访问和克隆需要已获授权的 GitHub 账号。

## 更新和上传源码

开始修改前，在工作区没有未提交改动时获取远程更新：

```bash
git pull --ff-only
```

完成修改后，检查并上传：

```bash
git status
git add -A
git diff --cached --stat
git commit -m "Describe this change"
git push
```

提交说明应描述本次实际改动。数据集、模型权重、特征缓存和实验输出由 `.gitignore` 排除；训练和测试得到的文件继续保留在本地。需要公开某一项实验结果时，应单独整理其来源、配置和可发布内容。

## 上传新版本压缩包

先提交并推送源码，再生成同一版本的源码包。例如下一版为 `v0.1.1`：

```bash
python scripts/package_release.py --output dist/MERC-GitHub-v0.1.1-source.zip
git tag -a v0.1.1 -m "MSE-IIPF v0.1.1"
git push origin v0.1.1
```

随后在 GitHub 的 **Releases → Draft a new release** 中选择该标签，填写版本说明，附上 ZIP 及其 `.sha256` 文件，再发布。每个新版本使用新的标签和文件名。

源码直接保存在 Git 仓库中，便于查看修改历史；ZIP 保存在 Release 中，便于下载。详细发布说明见 [RELEASING.md](RELEASING.md)。

## 在另一台电脑继续工作

复制 `git remote get-url origin` 输出的 URL，在另一台电脑登录有权限的 GitHub 账号后运行 `git clone URL`。进入克隆后的目录，按照 [README](../README.md#installation) 安装依赖。数据和权重需要下载或从已有本地目录复用。

如果 `git push` 提示需要登录，使用 Git Credential Manager 弹出的官方登录窗口，或运行 `gh auth login`（需先安装 GitHub CLI）。不要把访问令牌写入仓库文件、提交记录或远程 URL。
