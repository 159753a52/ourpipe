#!/usr/bin/env bash
set -euo pipefail
echo "== Basic info =="
echo "User: $USER  Host: $(hostname)  PWD: $(pwd)"
echo "OS: $(grep -E '^NAME=|^VERSION=' /etc/os-release 2>/dev/null | tr '\n' ' ' || echo 'unknown')"
echo "Kernel: $(uname -r)"
echo

echo "== Docker client =="
if command -v docker >/dev/null 2>&1; then
  echo "docker: FOUND ($(docker --version 2>/dev/null))"
else
  echo "docker: NOT FOUND"
fi
if command -v dockerd >/dev/null 2>&1; then
  echo "dockerd: FOUND ($(dockerd --version 2>/dev/null | head -n1))"
else
  echo "dockerd: NOT FOUND"
fi
echo

echo "== Daemon/socket =="
if [ -S /var/run/docker.sock ]; then
  ls -l /var/run/docker.sock
else
  echo "/var/run/docker.sock: not present"
fi
if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active --quiet docker && echo "systemd: docker service ACTIVE" || echo "systemd: docker service NOT active"
else
  echo "systemctl not available"
fi
echo "dockerd proc:"
ps -ef | grep -v grep | grep -E 'dockerd|dockerd' || echo "no dockerd process"
echo

echo "== Rootless Docker check (无 root 模式) =="
if command -v dockerd-rootless-setuptool.sh >/dev/null 2>&1; then
  echo "rootless tool: FOUND"
  dockerd-rootless-setuptool.sh check || true
else
  echo "rootless tool: NOT FOUND"
fi
echo

echo "== Group/membership =="
if getent group docker >/dev/null 2>&1; then
  echo "group 'docker' exists"
  id | grep -q "(docker)" && echo "you ARE in 'docker' group" || echo "you are NOT in 'docker' group"
else
  echo "group 'docker' does not exist"
fi
echo

echo "== Environment Modules =="
if command -v module >/dev/null 2>&1 || [ -n "${LMOD_CMD-}" ]; then
  module avail 2>/dev/null | egrep -i 'docker|podman|singularity|apptainer' || true
else
  echo "environment modules not available"
fi
echo

echo "== Podman (rootless OCI) =="
if command -v podman >/dev/null 2>&1; then podman --version; else echo "podman: NOT FOUND"; fi
echo

echo "== Remote contexts (if docker client exists) =="
if command -v docker >/dev/null 2>&1; then
  docker context ls || true
  echo "DOCKER_HOST=${DOCKER_HOST-}"
else
  echo "skip: no docker client"
fi
