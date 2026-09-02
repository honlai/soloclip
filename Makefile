# soloclip - the operations worth having a shorthand for. Paths and pgrep
#
# patterns are pinned here so they are not retyped by hand each time.
#
# Why this file exists: long runs need setsid, a known output path, and pgrep
# patterns that do not match themselves. All three have gone wrong here before -
# a liveness check that always said "running", a pkill that killed its own shell,
# and output sent to /dev/null so the first crash left no evidence.

# Whatever python is on PATH, i.e. the activated env. Override per machine in
# local.mk (gitignored) rather than editing this file.
PY      ?= python
-include local.mk
# Which list to work on. Every target honours it:  make up CFG=configs/talks.yaml
CFG     ?= configs/interviews.yaml
NAME    := $(notdir $(basename $(CFG)))
VAR     := var
SUP_OUT := $(VAR)/sup.$(NAME).out
WD_OUT  := $(VAR)/wd.$(NAME).out
STALE   := 600

# The character class matches as a regex but not as a literal, so the pattern
# does not find the very command line that contains it.
P_RUN  := bin/python -u -m solocli[p]
P_SUP  := bash tools/supervis[e].sh
P_WD   := bash tools/watchdo[g].sh

.PHONY: help up run watch status tail stop test sweep rescore rerender manifest clean-logs pairs host

help:
	@echo "up        start supervisor + watchdog in the background (detached)"
	@echo "          pick the list with CFG=, default $(CFG)"
	@echo "          e.g. make up CFG=configs/talks.yaml"
	@echo "status    liveness, throughput and progress, with timestamps"
	@echo "tail      follow the supervisor output"
	@echo "stop      stop the run and the watchers"
	@echo "manifest  per-video results table"
	@echo "test      unit tests"
	@echo "sweep     threshold sweep over cached measurements, no GPU"
	@echo "          e.g. make sweep YAW='30 35' SIZE='0.06 0.05'"
	@echo "rescore   apply new thresholds to the cache without re-detecting"
	@echo "rerender  rescore and re-render each video"

$(VAR):
	@mkdir -p $(VAR)

up: run watch
	@echo "started; make status"

# The liveness guard must NOT share a command line with the launch string:
# make runs each recipe line as one `sh -c "<whole line>"`, so a pgrep pattern
# and the `bash tools/supervise.sh ...` it guards would both sit in that
# shell's own argv - and pgrep -f would match the guard itself, reporting
# "already running" forever and never starting anything. Guard on a pidfile.
run: | $(VAR)
	@if [ -e $(VAR)/sup.pid ] && kill -0 $$(cat $(VAR)/sup.pid) 2>/dev/null; then \
		echo "supervisor already running (pid $$(cat $(VAR)/sup.pid))"; \
	else \
		nohup setsid bash tools/supervise.sh $(SUP_OUT) $(CFG) >/dev/null 2>&1 </dev/null & \
		echo $$! > $(VAR)/sup.pid; echo "supervisor started"; \
	fi

watch: | $(VAR)
	@if [ -e $(VAR)/wd.pid ] && kill -0 $$(cat $(VAR)/wd.pid) 2>/dev/null; then \
		echo "watchdog already running (pid $$(cat $(VAR)/wd.pid))"; \
	else \
		nohup setsid bash tools/watchdog.sh $(SUP_OUT) $(STALE) > $(WD_OUT) 2>&1 </dev/null & \
		echo $$! > $(VAR)/wd.pid; echo "watchdog started"; \
	fi

# Liveness and rate. A running total on its own is meaningless - it was once read
# as "3 clips in two hours" when the two figures were 20 minutes apart - so the
# time and the per-video cost always come with it.
status:
	@echo "now        $$(date '+%Y-%m-%d %H:%M:%S')"
	@printf "processes  run=%s supervisor=%s watchdog=%s\n" \
		"$$(pgrep -cf '$(P_RUN)')" "$$(pgrep -cf '$(P_SUP)')" "$$(pgrep -cf '$(P_WD)')"
	@L=$$(ls -t logs/*-run.log 2>/dev/null | head -1); \
		[ -n "$$L" ] && echo "log        $$(basename $$L), $$(( $$(date +%s) - $$(stat -c %Y $$L) ))s ago" || true
	@echo "clips      $$(ls out/*.mp4 2>/dev/null | wc -l)"
	@$(PY) tools/rate.py $(CFG) 2>/dev/null || $(PY) tools/rate.py

tail:
	@tail -f $(SUP_OUT)

stop:
	@pkill -f "$(P_SUP)" 2>/dev/null; pkill -f "$(P_WD)" 2>/dev/null; pkill -f "$(P_RUN)" 2>/dev/null; true
	@rm -f $(VAR)/sup.pid $(VAR)/wd.pid
	@echo "stopped"

test:
	@PYTHONPATH=src $(PY) tools/run_tests.py

YAW  ?= 30 35 40 45
SIZE ?= 0.06 0.05 0.04
sweep:
	@PYTHONPATH=src $(PY) tools/sweep.py --yaw $(YAW) --size $(SIZE)

rescore:
	@PYTHONPATH=src $(PY) -m soloclip asd --rescore

rerender:
	@bash tools/rerender.sh

manifest:
	@PYTHONPATH=src $(PY) -m soloclip status

clean-logs:
	@find logs -name '*.log' -size 0 -delete; echo "$$(ls logs/*.log 2>/dev/null | wc -l) log(s) left"

pairs:   ## write an audio twin of every finished clip (optional)
	@PYTHONPATH=src $(PY) -m soloclip -c $(CFG) run --help >/dev/null 2>&1 || true
	@PYTHONPATH=src $(PY) -m soloclip -c $(CFG) pair-audio

host:    ## find the recurring host from existing diarization results
	@PYTHONPATH=src $(PY) -m soloclip -c $(CFG) host
