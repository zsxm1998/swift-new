#!/bin/bash

# 初始化默认值
PARENT_DIR="/c22073"
USE_UID=false
PUBLIC_KEY=""

# 解析命名参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --name) USERNAME="$2"; shift ;;
    --home) PARENT_DIR="$2"; shift ;;
    --key) PUBLIC_KEY="$2"; shift ;;
    --uid) UID_VAL="$2"; USE_UID=true; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# 检查必需的参数
if [ -z "$USERNAME" ]; then
  echo "Usage: $0 --name <username> [--key <public_key>] [--home <parent_directory>] [--uid <uid>]"
  exit 1
fi

# 去掉 PARENT_DIR 结尾的斜杠（如果有的话）
PARENT_DIR=$(echo "$PARENT_DIR" | sed 's:/*$::')

# 根据是否传入 UID 决定如何创建用户
if [ "$USE_UID" = true ]; then
  # 先创建用户组
  groupadd -g "$UID_VAL" "$USERNAME"
  # 使用指定的 UID 创建用户，并指定主组为刚创建的同名组
  useradd -u "$UID_VAL" -g "$USERNAME" -d "$PARENT_DIR/$USERNAME" -m "$USERNAME" -s /bin/bash
else
  # 不指定 UID 和 GID 创建用户
  useradd -d "$PARENT_DIR/$USERNAME" -m "$USERNAME" -s /bin/bash
fi

# 仅在提供 --key 时才配置 SSH
if [ -n "$PUBLIC_KEY" ]; then
  # 创建 .ssh 目录，并设置权限
  mkdir -p "$PARENT_DIR/$USERNAME/.ssh"
  chmod 700 "$PARENT_DIR/$USERNAME/.ssh"

  # 将公钥写入 authorized_keys，并设置权限
  echo "$PUBLIC_KEY" > "$PARENT_DIR/$USERNAME/.ssh/authorized_keys"
  chmod 600 "$PARENT_DIR/$USERNAME/.ssh/authorized_keys"

  # 修改用户的 .ssh 目录权限
  chown -R "$USERNAME:$USERNAME" "$PARENT_DIR/$USERNAME/.ssh"
fi

echo "User $USERNAME created with home directory at $PARENT_DIR/$USERNAME"
