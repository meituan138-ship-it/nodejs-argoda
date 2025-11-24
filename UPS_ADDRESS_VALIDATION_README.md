# UPS 地址验证工具

使用UPS Address Validation API和OAuth认证的Python地址验证工具。

## 功能特性

- ✅ 使用OAuth 2.0进行身份认证（UPS新的认证方式）
- ✅ 支持美国私人和商业地址验证
- ✅ 自动token管理和刷新
- ✅ 内置Google、Apple、Microsoft、Amazon等大公司地址测试
- ✅ 详细的验证结果展示
- ✅ 错误处理和重试机制

## 环境要求

- Python 3.6+
- requests库

## 安装依赖

```bash
pip install -r requirements.txt
```

或者直接安装：

```bash
pip install requests
```

## 使用方法

### 1. 直接运行测试

脚本已经配置好了UPS API凭证，直接运行即可测试大公司地址：

```bash
python ups_address_validation.py
```

### 2. 作为模块使用

```python
from ups_address_validation import UPSAddressValidator

# 初始化验证器
validator = UPSAddressValidator(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# 验证地址
result = validator.validate_address(
    address_lines=["1600 Amphitheatre Parkway"],
    city="Mountain View",
    state="CA",
    postal_code="94043"
)

# 打印结果
validator.print_validation_result(result, "Google")
```

### 3. 自定义地址验证

```python
from ups_address_validation import UPSAddressValidator

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"

validator = UPSAddressValidator(CLIENT_ID, CLIENT_SECRET)

# 验证自定义地址
result = validator.validate_address(
    address_lines=["123 Main Street", "Apt 4B"],
    city="New York",
    state="NY",
    postal_code="10001"
)

validator.print_validation_result(result)
```

## API凭证配置

当前配置的UPS API凭证：

- **客户识别码**: jrv1UbCVu8WY7AwzbkmC6y1r9AmFDYHiChbCFxSV3Zp2ejIP
- **客户密钥**: T15iAJ3WESj7jNdWfi9VutfZ5O55SqQSbQsQ2vaMStbVrt1F4PYBKvEkxPeCLgwy
- **发行日期**: 08/14/2025

已订阅的API产品：
- ✅ Rating
- ✅ Address Validation
- ✅ Authorization (OAuth)
- ✅ Tracking
- ✅ Locator
- ✅ Shipping

## 输出示例

```
============================================================
UPS 地址验证工具
使用 OAuth 认证和 Address Validation API
============================================================

步骤 1: 测试 OAuth 认证
------------------------------------------------------------
正在获取新的OAuth访问令牌...
✓ 成功获取访问令牌 (有效期: 3600秒)

步骤 2: 测试大公司地址验证
------------------------------------------------------------

正在验证地址: 1600 Amphitheatre Parkway, Mountain View, CA 94043

============================================================
公司: Google (Googleplex 总部)
============================================================

✓ 找到 1 个匹配地址:

候选地址 #1:
  地址质量: Commercial
  地址: 1600 AMPHITHEATRE PKWY
  城市: MOUNTAIN VIEW
  州: CA
  邮编: 94043-1351
  国家: US
```

## 核心类和方法

### UPSAddressValidator

主验证器类，提供以下方法：

#### `__init__(client_id, client_secret)`
初始化验证器
- **client_id**: UPS客户识别码
- **client_secret**: UPS客户密钥

#### `get_oauth_token()`
获取OAuth访问令牌
- 自动管理token过期和刷新
- 返回有效的访问令牌

#### `validate_address(address_lines, city, state, postal_code, country_code="US")`
验证地址
- **address_lines**: 地址行列表
- **city**: 城市
- **state**: 州代码
- **postal_code**: 邮政编码
- **country_code**: 国家代码（默认US）
- 返回验证结果字典

#### `print_validation_result(result, company_name="")`
格式化打印验证结果
- **result**: 验证结果字典
- **company_name**: 公司名称（可选）

## API端点

- **OAuth认证**: https://onlinetools.ups.com/security/v1/oauth/token
- **地址验证**: https://onlinetools.ups.com/api/addressvalidation/v1/3

## 注意事项

1. **Token管理**: 访问令牌会自动缓存和刷新，有效期约1小时
2. **地址格式**: 地址行可以是单个字符串或字符串列表
3. **国家代码**: 目前配置为美国地址验证（US），可根据需要修改
4. **错误处理**: 所有API调用都包含错误处理和详细的错误信息输出

## 测试地址

脚本内置了以下公司的总部地址用于测试：

1. **Google** - 1600 Amphitheatre Parkway, Mountain View, CA 94043
2. **Apple** - One Apple Park Way, Cupertino, CA 95014
3. **Microsoft** - One Microsoft Way, Redmond, WA 98052
4. **Amazon** - 410 Terry Ave N, Seattle, WA 98109

## 故障排查

### OAuth认证失败
- 检查CLIENT_ID和CLIENT_SECRET是否正确
- 确认API凭证未过期（发行日期: 08/14/2025）
- 检查网络连接

### 地址验证失败
- 确保OAuth认证成功
- 检查地址格式是否正确
- 验证州代码和邮政编码是否匹配

### 网络超时
- 增加timeout参数值
- 检查防火墙设置
- 确认UPS API服务可访问

## 参考文档

- [UPS OAuth认证文档](https://developer.ups.com/tag/OAuth-Auth-Code?loc=zh_CN)
- [UPS Address Validation API文档](https://developer.ups.com/tag/Address-Validation?loc=en_US)

## 许可证

本工具仅供学习和测试使用。
