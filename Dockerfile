FROM ghcr.io/synaptics-synap/toolkit:3.1.0
USER root
RUN echo '=== SYNAP IMAGE PATH ===' && echo "$PATH" && echo '=== SYNAP FILES ===' && (find / -maxdepth 6 \( -iname 'synap' -o -iname 'synap_convert*' -o -iname '*entrypoint*' -o -iname 'pysynap' \) 2>/dev/null | head -300 || true) && echo '=== PYTHON/VENV ===' && (find / -maxdepth 5 -type f \( -name 'python' -o -name 'python3' \) 2>/dev/null | head -100 || true) && echo '=== ROOT DIRS ===' && ls -la /
ENTRYPOINT []
COPY runner.sh /runner.sh
RUN chmod +x /runner.sh
CMD ["/runner.sh"]
