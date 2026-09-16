FROM ghcr.io/synaptics-synap/toolkit:3.1.0
USER root
ENTRYPOINT []
COPY runner.sh /runner.sh
RUN chmod +x /runner.sh
CMD ["/runner.sh"]
