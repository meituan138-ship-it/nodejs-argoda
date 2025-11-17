#!/bin/bash

echo "=========================================="
echo "n8n API 诊断和修复脚本"
echo "=========================================="
echo ""

# 1. 检查 Docker 容器状态
echo "1. 检查 Docker 容器..."
docker ps -a | grep -E "n8n|envoy"
echo ""

# 2. 查看 n8n 容器日志
echo "2. n8n 容器最新日志..."
N8N_CONTAINER=$(docker ps -q --filter "name=n8n" | head -1)
if [ ! -z "$N8N_CONTAINER" ]; then
    docker logs --tail 30 $N8N_CONTAINER
else
    echo "未找到 n8n 容器"
fi
echo ""

# 3. 检查 n8n 环境变量
echo "3. n8n 环境变量配置..."
if [ ! -z "$N8N_CONTAINER" ]; then
    docker exec $N8N_CONTAINER printenv | grep -E "N8N_|PUBLIC_API" | sort
else
    echo "未找到 n8n 容器"
fi
echo ""

# 4. 检查 Envoy 配置
echo "4. 检查 Envoy 配置..."
ENVOY_CONTAINER=$(docker ps -q --filter "name=envoy" | head -1)
if [ ! -z "$ENVOY_CONTAINER" ]; then
    echo "Envoy 容器 ID: $ENVOY_CONTAINER"
    docker exec $ENVOY_CONTAINER cat /etc/envoy/envoy.yaml 2>/dev/null || \
    docker exec $ENVOY_CONTAINER cat /etc/envoy/envoy.json 2>/dev/null || \
    echo "无法读取 Envoy 配置"
else
    echo "未找到 Envoy 容器"
    # 检查主机上的 Envoy 配置
    find /etc -name "*envoy*" 2>/dev/null
fi
echo ""

# 5. 检查网络监听端口
echo "5. 检查端口监听..."
netstat -tlnp | grep -E ":5678|:8080|:443"
echo ""

# 6. 测试本地访问
echo "6. 测试从服务器内部访问 n8n..."
curl -s http://localhost:5678/healthz && echo "健康检查: OK" || echo "健康检查: FAILED"
curl -s http://localhost:5678/api/v1/workflows -H "X-N8N-API-KEY: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZmQ3OWY4Yi1kZmRiLTQxNjAtYTg2YS1mY2M5ZDVlNzM3MjciLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzYzMzkzNTczLCJleHAiOjE3NzEwODQ4MDB9.2kdxWqf0ASMTQox_ZABlZVij7_rqYOS3mWVxZOM6ZIE" && echo "API 访问: OK" || echo "API 访问: FAILED"
echo ""

# 7. 检查 docker-compose 配置
echo "7. 检查 docker-compose 配置..."
find /root /home -name "docker-compose.yml" -o -name "docker-compose.yaml" 2>/dev/null | head -5
echo ""

echo "=========================================="
echo "诊断完成！请将以上输出发送给我进行分析"
echo "=========================================="
