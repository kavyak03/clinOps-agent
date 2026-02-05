# 🚀 Windows + VS Code + WSL2 + Docker Setup Guide

This guide helps **Windows users** clone and run this repo smoothly using:

- WSL2 (Ubuntu)
- VS Code (Remote – WSL)
- Docker Desktop
- GitHub (SSH or Token auth)

If you follow these steps, everything should "just work" without fighting environment issues.

---

# ✅ 0. Requirements

Install:

- WSL2
- Ubuntu (WSL distro)
- Visual Studio Code
- VS Code extension: **Remote – WSL**
- Docker Desktop

---

# ✅ 1. Install / Verify WSL2 + Ubuntu

Open **PowerShell**:

```powershell
wsl -l -v
```

Expected:
```
NAME              STATE           VERSION
Ubuntu            Running         2
docker-desktop    Running         2
```

If Ubuntu is missing:

```powershell
wsl --install -d Ubuntu
```

Set WSL2 as default:

```powershell
wsl --set-default-version 2
wsl --set-default Ubuntu
```

Open Ubuntu:

```powershell
wsl
```

---

# ✅ 2. Clone Repo Inside WSL (IMPORTANT)

Do NOT work inside `/mnt/c` or `/mnt/e` for development (slow builds).

Instead:

```bash
mkdir -p ~/projects
cd ~/projects
```

Clone:

## Option A — SSH (recommended)
```bash
git clone git@github.com:<username>/<repo>.git
```

## Option B — HTTPS + token
```bash
git clone https://github.com/<username>/<repo>.git
```
When prompted:
- Username = GitHub username
- Password = **Personal Access Token (NOT password)**

---

# ✅ 3. Open in VS Code

From WSL terminal:

```bash
code .
```

OR

VS Code → Ctrl+Shift+P → "WSL: Connect to WSL"

Bottom-left should show:

```
WSL: Ubuntu
```

---

# ✅ 4. GitHub Authentication Setup

GitHub passwords do NOT work for git commands.

## Recommended: SSH setup

Inside WSL:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy key → GitHub → Settings → SSH keys → Add

After this, git will never ask for passwords again.

---

# ✅ 5. Docker Setup (WSL2)

Install **Docker Desktop (Windows)**.

Then:

Docker Desktop → Settings → WSL Integration → Enable Ubuntu → Restart

---

## Test Docker inside WSL

```bash
docker version
docker ps
```

If docker is not found:

```bash
sudo apt update
sudo apt install -y docker.io
```

---

# ✅ 6. Build Project

From repo root:

```bash
docker build --no-cache -t clinrag:test .
```

---

# ✅ 7. Common Fixes

## Docker not found
Install docker CLI:
```bash
sudo apt install docker.io
```

## Docker cannot connect
Restart:
```powershell
wsl --shutdown
```
Restart Docker Desktop.

## Using WSL1
Docker only works with WSL2. Convert or switch to Ubuntu.

## VS Code opens wrong environment
Ensure bottom-left says:
```
WSL: Ubuntu
```

---

# ✅ 8. Quick Sanity Check

PowerShell:
```powershell
wsl -l -v
```

WSL:
```bash
uname -a
which docker
docker version
```

If these work → you’re ready 🎉

---

# 💡 Recommended Workflow (daily)

```bash
wsl
cd ~/projects/<repo>
code .
docker build -t clinrag:test .
```