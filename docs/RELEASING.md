# GitHub 发布与压缩包上传

仓库建立后的日常同步、提交和版本管理见 [GIT_WORKFLOW.md](GIT_WORKFLOW.md)。

## 推荐操作

1. 解压源码包，进入 `MERC-GitHub`。将这一目录中的文件作为仓库根目录内容，确保 `README.md` 位于仓库根目录。
2. 在 GitHub 创建新仓库。可使用 Git 推送源码，或使用网页上传解压后的文件；上传时保留 `.github`、`.gitignore` 等隐藏文件。
3. 在仓库的 **Releases → Draft a new release** 中创建版本，例如 `v0.1.0`，附上本次源码 ZIP 和 `.sha256` 校验文件，再发布。[GitHub 官方 Release 操作说明](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

GitHub 会识别仓库中的根目录 README。仅将 ZIP 当作普通仓库文件上传，不会把包内源码自动展开成可浏览项目，也不会展示包内 README。要同时具备项目首页和压缩包下载入口，应提交解压后的源码，再附加 Release 压缩包。[GitHub README 说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)

## 重新打包

```bash
python scripts/package_release.py --output dist/MERC-GitHub-source.zip
```

打包依据文件白名单，不会将 `data/`、`weights/`、`cache/`、`runs/`、`figures/`、`tables/` 等运行目录带入。ZIP 内含 `SHA256SUMS.txt`，ZIP 同级生成 `.zip.sha256`。脚本拒绝覆盖已有同名 ZIP，需要使用新文件名。

PowerShell 可核验整个压缩包：

```powershell
Get-FileHash -Algorithm SHA256 dist/MERC-GitHub-source.zip
```

## 发布信息

源码包已具备运行入口、说明及测试配置。仓库所有者、论文作者、正式 DOI 和原创代码许可证无法从目录名可靠推断，发布者应根据实际情况填写。不能把保留的 ALBEF 许可证当作整个项目的许可声明。

本 README 采用“项目介绍—环境安装—数据准备—训练—测试—输出—目录—引用与维护”的组织方式，参考了 [GitHub 官方 README 指引](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)以及 [MERTools](https://github.com/zeroQiaoba/MERTools)、[CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) 等实际学术仓库。没有复制其结果或为本项目生成虚构的星标、徽章、作者、DOI。
