# Linux systemd 示例

这些文件只负责部署独立视觉识别服务，不部署 FileService。

推荐目录：

```text
/opt/crosscart/visual-search/service      Python 服务代码
/opt/crosscart/visual-search/venv         Python 虚拟环境
/var/lib/crosscart/visual-search          SQLite 索引库和模型缓存
/var/lib/crosscart/upload                 FileService upload 原图目录，可改成真实路径
/etc/crosscart/visual-search.env          环境变量
```

安装服务：

```bash
sudo cp crosscart-visual-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crosscart-visual-search
sudo systemctl status crosscart-visual-search
journalctl -u crosscart-visual-search -f
```

服务默认监听 `127.0.0.1:8100`。管理台地址：

```text
http://127.0.0.1:8100/admin
```
