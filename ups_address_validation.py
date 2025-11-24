#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPS Address Validation Tool
使用OAuth认证和Address Validation API进行地址验证
"""

import os
import requests
import json
import base64
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# 尝试导入python-dotenv（可选）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载.env文件中的环境变量
except ImportError:
    pass  # 如果没有安装python-dotenv，继续使用系统环境变量


class UPSAddressValidator:
    """UPS地址验证类"""

    # UPS API端点
    OAUTH_URL = "https://onlinetools.ups.com/security/v1/oauth/token"
    ADDRESS_VALIDATION_URL = "https://onlinetools.ups.com/api/addressvalidation/v1/3"

    def __init__(self, client_id: str, client_secret: str):
        """
        初始化UPS地址验证器

        Args:
            client_id: UPS客户识别码
            client_secret: UPS客户密钥
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = None

    def get_oauth_token(self) -> str:
        """
        获取OAuth访问令牌

        Returns:
            访问令牌字符串
        """
        # 检查是否已有有效token
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                print("使用现有的访问令牌")
                return self.access_token

        print("正在获取新的OAuth访问令牌...")

        # 创建Basic Auth头
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded_credentials}"
        }

        data = {
            "grant_type": "client_credentials"
        }

        try:
            response = requests.post(
                self.OAUTH_URL,
                headers=headers,
                data=data,
                timeout=30
            )
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data.get("access_token")
            expires_in = int(token_data.get("expires_in", 3600))

            # 设置token过期时间（提前5分钟刷新）
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

            print(f"✓ 成功获取访问令牌 (有效期: {expires_in}秒)")
            return self.access_token

        except requests.exceptions.RequestException as e:
            print(f"✗ OAuth认证失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            raise

    def validate_address(
        self,
        address_lines: list,
        city: str,
        state: str,
        postal_code: str,
        country_code: str = "US"
    ) -> Dict:
        """
        验证地址

        Args:
            address_lines: 地址行列表（如街道地址）
            city: 城市
            state: 州/省
            postal_code: 邮政编码
            country_code: 国家代码（默认US）

        Returns:
            验证结果字典
        """
        # 获取访问令牌
        token = self.get_oauth_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "transId": f"Address-Validation-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transactionSrc": "AddressValidationTool"
        }

        # 构建请求体
        payload = {
            "XAVRequest": {
                "AddressKeyFormat": {
                    "ConsigneeName": "",
                    "AddressLine": address_lines if isinstance(address_lines, list) else [address_lines],
                    "Region": f"{city},{state},{postal_code}",
                    "CountryCode": country_code
                }
            }
        }

        print(f"\n正在验证地址: {', '.join(address_lines)}, {city}, {state} {postal_code}")

        try:
            response = requests.post(
                self.ADDRESS_VALIDATION_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result

        except requests.exceptions.RequestException as e:
            print(f"✗ 地址验证失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            raise

    def print_validation_result(self, result: Dict, company_name: str = ""):
        """
        格式化打印验证结果

        Args:
            result: 验证结果字典
            company_name: 公司名称（用于显示）
        """
        print("\n" + "="*60)
        if company_name:
            print(f"公司: {company_name}")
        print("="*60)

        try:
            xav_response = result.get("XAVResponse", {})

            # 检查是否有有效候选地址
            candidate = xav_response.get("Candidate")

            if candidate:
                # 处理单个候选地址或多个候选地址
                candidates = candidate if isinstance(candidate, list) else [candidate]

                print(f"\n✓ 找到 {len(candidates)} 个匹配地址:\n")

                for idx, addr in enumerate(candidates, 1):
                    print(f"候选地址 #{idx}:")

                    # 地址质量
                    quality = addr.get("AddressClassification", {}).get("Description", "未知")
                    print(f"  地址质量: {quality}")

                    # 详细地址
                    address_key = addr.get("AddressKeyFormat", {})
                    address_lines = address_key.get("AddressLine", [])

                    if isinstance(address_lines, list):
                        for line in address_lines:
                            print(f"  地址: {line}")
                    else:
                        print(f"  地址: {address_lines}")

                    # 城市、州、邮编
                    city = address_key.get("PoliticalDivision2", "")
                    state = address_key.get("PoliticalDivision1", "")
                    postal_code = address_key.get("PostcodePrimaryLow", "")
                    postal_ext = address_key.get("PostcodeExtendedLow", "")
                    country = address_key.get("CountryCode", "")

                    full_postal = f"{postal_code}-{postal_ext}" if postal_ext else postal_code

                    print(f"  城市: {city}")
                    print(f"  州: {state}")
                    print(f"  邮编: {full_postal}")
                    print(f"  国家: {country}")

                    print()

            # 检查是否有错误或无匹配
            elif "NoCandidatesIndicator" in xav_response:
                print("\n✗ 没有找到匹配的地址")

            elif "AmbiguousAddressIndicator" in xav_response:
                print("\n⚠ 地址信息不明确，无法验证")

            # 显示原始响应（仅在调试时）
            # print("\n原始响应:")
            # print(json.dumps(result, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"\n✗ 解析验证结果时出错: {e}")
            print("\n原始响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))


def test_company_addresses(validator: UPSAddressValidator):
    """
    测试大公司地址

    Args:
        validator: UPS地址验证器实例
    """
    print("\n" + "="*60)
    print("开始测试大公司地址验证")
    print("="*60)

    # 测试地址列表
    test_addresses = [
        {
            "company": "Google (Googleplex 总部)",
            "address_lines": ["1600 Amphitheatre Parkway"],
            "city": "Mountain View",
            "state": "CA",
            "postal_code": "94043"
        },
        {
            "company": "Apple (Apple Park 总部)",
            "address_lines": ["One Apple Park Way"],
            "city": "Cupertino",
            "state": "CA",
            "postal_code": "95014"
        },
        {
            "company": "Microsoft (Redmond 总部)",
            "address_lines": ["One Microsoft Way"],
            "city": "Redmond",
            "state": "WA",
            "postal_code": "98052"
        },
        {
            "company": "Amazon (Seattle 总部)",
            "address_lines": ["410 Terry Ave N"],
            "city": "Seattle",
            "state": "WA",
            "postal_code": "98109"
        }
    ]

    for addr_info in test_addresses:
        try:
            result = validator.validate_address(
                address_lines=addr_info["address_lines"],
                city=addr_info["city"],
                state=addr_info["state"],
                postal_code=addr_info["postal_code"]
            )

            validator.print_validation_result(result, addr_info["company"])

        except Exception as e:
            print(f"\n✗ 验证 {addr_info['company']} 地址时出错: {e}")

        print("\n" + "-"*60)


def main():
    """主函数"""
    print("="*60)
    print("UPS 地址验证工具")
    print("使用 OAuth 认证和 Address Validation API")
    print("="*60)

    # 从环境变量读取UPS API凭证
    CLIENT_ID = os.getenv("UPS_CLIENT_ID")
    CLIENT_SECRET = os.getenv("UPS_CLIENT_SECRET")

    # 检查凭证是否存在
    if not CLIENT_ID or not CLIENT_SECRET:
        print("\n❌ 错误: 未找到UPS API凭证!")
        print("\n请设置以下环境变量:")
        print("  - UPS_CLIENT_ID: 您的UPS客户识别码")
        print("  - UPS_CLIENT_SECRET: 您的UPS客户密钥")
        print("\n方法1: 使用.env文件 (推荐)")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 在 .env 文件中填入您的凭证")
        print("  3. 运行: pip install python-dotenv")
        print("\n方法2: 直接设置环境变量")
        print("  export UPS_CLIENT_ID='your_client_id'")
        print("  export UPS_CLIENT_SECRET='your_client_secret'")
        return 1

    # 创建验证器实例
    validator = UPSAddressValidator(CLIENT_ID, CLIENT_SECRET)

    try:
        # 测试OAuth认证
        print("\n步骤 1: 测试 OAuth 认证")
        print("-"*60)
        validator.get_oauth_token()

        # 测试大公司地址
        print("\n步骤 2: 测试大公司地址验证")
        print("-"*60)
        test_company_addresses(validator)

        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 程序执行出错: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
