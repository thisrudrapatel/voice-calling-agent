# Asterisk 20 from Ubuntu 24.04 (includes app_audiosocket + chan_pjsip).
FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends asterisk \
 && rm -rf /var/lib/apt/lists/*
COPY pjsip.conf extensions.conf rtp.conf modules.conf /etc/asterisk/
CMD ["asterisk", "-f", "-vvv"]
