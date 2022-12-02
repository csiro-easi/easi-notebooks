#!/bin/bash
#
# https://stackoverflow.com/a/3464399
#
#HOOK_NAMES="applypatch-msg pre-applypatch post-applypatch pre-commit prepare-commit-msg commit-msg post-commit pre-rebase post-checkout post-merge pre-receive update post-receive post-update pre-auto-gc"
HOOK_NAMES="pre-commit"

TOP=$(git rev-parse --show-toplevel)
BIN_DIR=$TOP/bin
HOOK_DIR=$TOP/.git/hooks

for hook in $HOOK_NAMES; do
    # If the hook already exists, is executable, and is not a symlink
    if [ ! -h $HOOK_DIR/$hook -a -x $HOOK_DIR/$hook ]; then
        mv $HOOK_DIR/$hook $HOOK_DIR/$hook.local
    fi
    # Create a symlink, overwriting the file if it exists
    ln -s -f $TOP/bin/$hook $HOOK_DIR/$hook
    chmod +x $TOP/bin/$hook
    echo Applied $hook hook: `ls -l $HOOK_DIR/$hook | sed -E 's/^.+\.git/\.git/'`
done
