"""
项目自动同步 GitHub 脚本。

功能：
- 定期检查工作区是否有变更
- 有变更时自动 add、commit（附带时间戳）并 push 到 origin
- 无变更时静默等待

用法：
    python auto_sync.py

默认每 86400 秒（1 天）检查一次，可通过环境变量 SYNC_INTERVAL 修改：
    set SYNC_INTERVAL=3600
    python auto_sync.py

首次使用前请确保：
1. 已在 GitHub 创建仓库
2. 已添加远程地址：git remote add origin <你的仓库地址>
3. 本地已配置好 Git 身份（user.name / user.email）
4. 已配置好 GitHub 认证（SSH 密钥、PAT Token 或 GitHub CLI）
"""

import os
import subprocess
import sys
import time
from datetime import datetime


def run_git(args, check=True):
    """执行 git 命令并返回输出。"""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        check=check,
    )
    return result


def has_remote():
    """检查是否配置了远程仓库 origin。"""
    result = run_git(["remote", "get-url", "origin"], check=False)
    return result.returncode == 0 and result.stdout.strip()


def has_changes():
    """检查工作区是否有未提交的变更。"""
    result = run_git(["status", "--porcelain"])
    return bool(result.stdout.strip())


def get_current_branch():
    """获取当前分支名。"""
    result = run_git(["branch", "--show-current"])
    return result.stdout.strip()


def sync():
    """执行一次同步检查。"""
    if not has_remote():
        print("[错误] 未配置远程仓库 origin。")
        print("请先执行：git remote add origin <你的 GitHub 仓库地址>")
        return False

    if not has_changes():
        return True  # 没有变更，无需操作

    branch = get_current_branch()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"auto sync: {now}"

    print(f"[{now}] 检测到变更，开始同步...")

    run_git(["add", "."])
    run_git(["commit", "-m", message])
    run_git(["push", "origin", branch])

    print(f"[{now}] 已推送到 origin/{branch}")
    return True


def main():
    interval = int(os.environ.get("SYNC_INTERVAL", "86400"))
    print(f"自动同步已启动，每 {interval} 秒检查一次（约 {interval / 3600:.1f} 小时）。按 Ctrl+C 停止。")
    print(f"当前分支: {get_current_branch()}")

    if not has_remote():
        print("\n[提示] 尚未配置 GitHub 远程仓库。")
        print("请在 GitHub 创建仓库后执行：")
        print("  git remote add origin https://github.com/<用户名>/<仓库名>.git")
        print("  git push -u origin master\n")

    try:
        while True:
            sync()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n自动同步已停止。")
        sys.exit(0)


if __name__ == "__main__":
    main()
