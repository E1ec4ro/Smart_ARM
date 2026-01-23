#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/log/supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

