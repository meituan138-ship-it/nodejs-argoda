# 安全公告 (Security Advisory)

## ⚠️ 重要安全提醒

### API凭证泄露问题

在项目的早期提交中（commit 2038c05），UPS API凭证被硬编码在源代码中并提交到了Git仓库。这些凭证已经被推送到GitHub公开仓库。

**受影响的文件：**
- `ups_address_validation.py` (commit 2038c05)
- `UPS_ADDRESS_VALIDATION_README.md` (commit 2038c05)

**泄露的凭证信息：**
- UPS Client ID
- UPS Client Secret

## 🔒 已采取的安全措施

### 1. 代码修复 (commit 7412ad7)

我们已经在后续提交中修复了这个安全问题：

- ✅ 从源代码中移除了所有硬编码的凭证
- ✅ 实现了环境变量配置方式
- ✅ 添加了 `.gitignore` 防止 `.env` 文件被提交
- ✅ 创建了 `.env.example` 作为配置模板
- ✅ 更新了文档，移除了敏感信息
- ✅ 添加了安全最佳实践指南

### 2. 建议的补救措施

**⚠️ 强烈建议立即采取以下行动：**

#### 步骤 1: 撤销泄露的凭证

1. 登录 [UPS Developer Portal](https://developer.ups.com/)
2. 找到相关的应用程序
3. **立即撤销当前的 API 凭证**
4. 生成新的 Client ID 和 Client Secret
5. 将新凭证保存在安全的地方（如密码管理器）

#### 步骤 2: 配置新凭证

使用新生成的凭证：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入新的凭证
nano .env
```

在 `.env` 文件中：
```
UPS_CLIENT_ID=your_new_client_id_here
UPS_CLIENT_SECRET=your_new_client_secret_here
```

#### 步骤 3: 验证安全配置

```bash
# 确认 .env 文件不在Git跟踪中
git status

# .env 应该不会出现在未跟踪文件列表中
# 如果出现，说明.gitignore配置正确

# 测试程序是否正常工作
python3 ups_address_validation.py
```

#### 步骤 4: （可选）清理Git历史

如果需要从Git历史中完全移除敏感信息：

**⚠️ 警告：这将重写Git历史，会影响所有克隆此仓库的用户**

```bash
# 安装 git-filter-repo
pip install git-filter-repo

# 备份仓库
cp -r .git .git.backup

# 从历史中移除包含凭证的提交（谨慎操作）
# 选项1: 移除特定文件的历史
git filter-repo --path ups_address_validation.py --invert-paths --force
git filter-repo --path UPS_ADDRESS_VALIDATION_README.md --invert-paths --force

# 选项2: 使用 BFG Repo-Cleaner (更快)
# 下载: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files ups_address_validation.py
java -jar bfg.jar --delete-files UPS_ADDRESS_VALIDATION_README.md
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# 强制推送新历史
git push origin --force --all
```

**重要提示：** 即使清理了Git历史，泄露的凭证可能已经被缓存或索引。**必须撤销旧凭证并生成新凭证。**

## 📋 安全最佳实践

### 对于开发者

1. **永远不要**将 API 密钥、密码或其他敏感信息提交到Git
2. **始终使用**环境变量或密钥管理服务
3. **配置** `.gitignore` 忽略包含敏感信息的文件
4. **使用** pre-commit hooks 检测敏感信息
5. **定期轮换** API 密钥和凭证
6. **启用** 2FA（双因素认证）保护账户

### Pre-commit Hook 示例

在 `.git/hooks/pre-commit` 中添加：

```bash
#!/bin/bash
# 检测是否有敏感信息被提交

if git diff --cached | grep -E "(client_id|client_secret|api_key|password|token)" -i; then
    echo "❌ 错误: 检测到可能的敏感信息!"
    echo "请检查您的提交内容，确保没有包含API密钥或密码。"
    exit 1
fi
```

使其可执行：
```bash
chmod +x .git/hooks/pre-commit
```

### 推荐工具

- **git-secrets**: 防止提交密钥和凭证
  ```bash
  git secrets --install
  git secrets --register-aws
  ```

- **truffleHog**: 扫描Git历史中的敏感信息
  ```bash
  pip install truffleHog
  truffleHog --regex --entropy=True .
  ```

- **GitGuardian**: 实时监控Git提交中的密钥泄露
  - 网站: https://www.gitguardian.com/

## 📞 报告安全问题

如果您发现本项目的安全问题，请：

1. **不要**在公开的 issue 中报告
2. 通过私密方式联系项目维护者
3. 等待响应和修复后再公开

## 🔄 更新日志

### 2024-11-24
- **发现**: API凭证被硬编码在源代码中 (commit 2038c05)
- **修复**: 实现环境变量配置，移除硬编码凭证 (commit 7412ad7)
- **建议**: 立即撤销并重新生成新的API凭证

## 📚 参考资料

- [OWASP Top 10 - A07 Identification and Authentication Failures](https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/)
- [GitHub - Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [UPS Developer Portal - Security Best Practices](https://developer.ups.com/)

## ⚖️ 免责声明

本安全公告仅供信息目的。使用本项目及相关API服务时，请遵守UPS的服务条款和隐私政策。项目维护者不对因凭证泄露造成的任何损失负责。
