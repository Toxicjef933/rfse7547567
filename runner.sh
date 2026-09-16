#!/bin/sh
set -eu
echo '=== INSPECAO DA IMAGEM SYNAP ==='
echo "PATH=$PATH"
echo '--- python ---'
which python3 || true
python3 - <<'PY'
import sys, os
print('python:', sys.executable)
print('sys.path:')
for p in sys.path: print(' ', p)
PY

echo '--- procurando executaveis/arquivos SyNAP ---'
find / -maxdepth 6 \( -iname 'synap' -o -iname 'synap_convert*' -o -iname '*entrypoint*' -o -iname 'pysynap' \) 2>/dev/null | head -300 || true

echo '--- /usr/local/bin ---'
ls -la /usr/local/bin 2>/dev/null | head -200 || true
echo '--- /opt ---'
find /opt -maxdepth 4 -type f 2>/dev/null | head -300 || true
echo '--- /app /workspace /toolkit ---'
find /app /workspace /toolkit -maxdepth 4 -type f 2>/dev/null | head -300 || true

echo '=== FIM INSPECAO ==='
sleep 120
