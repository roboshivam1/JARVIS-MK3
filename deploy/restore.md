# Restoring JARVIS from a backup

A backup that has not been restored is a rumour. Run this once now, on
a copy, so you know it works before you need it.

## What a backup contains

    data/backups/2026-08-02T04-00/
      core.db            everything JARVIS knows
      artifacts.tar.gz   every file it has produced

Not included: `.env`. Keep your API keys in a password manager - an
off-site copy of them is a worse problem than losing them.

## Verifying a backup (do this now)

    BACKUP=data/backups/$(ls data/backups | tail -1)

    # Does the database open, and does it have your history?
    sqlite3 $BACKUP/core.db "PRAGMA integrity_check;"
    sqlite3 $BACKUP/core.db "SELECT COUNT(*) FROM events;"
    sqlite3 $BACKUP/core.db "SELECT COUNT(*) FROM facts WHERE status='active';"
    sqlite3 $BACKUP/core.db "SELECT content FROM profile_doc ORDER BY id DESC LIMIT 1;"

    # Does the archive open, and does it hold what you expect?
    tar -tzf $BACKUP/artifacts.tar.gz | head -20
    tar -tzf $BACKUP/artifacts.tar.gz | wc -l

If integrity_check says anything but `ok`, the backup is bad and the
job needs looking at.

## Full restore

On a machine with the code checked out and `.env` in place:

    # 1. Stop the Core if it is running.

    # 2. Move the current data aside rather than deleting it - if the
    #    restore goes wrong you want the option to go back.
    mv data data.before-restore

    mkdir -p data
    cp $BACKUP/core.db data/core.db
    tar -xzf $BACKUP/artifacts.tar.gz -C data

    # 3. Start. Migrations run automatically; a backup from an older
    #    build is brought forward on boot.
    uv run python -m jarvis.core

    # 4. Rebuild the browsable artifact tree, which is not archived
    #    (it is only symlinks into files already restored).
    #    See the rebuild_all snippet in the artifact_links module.

    # 5. Confirm from Telegram: /status, then ask about something only
    #    the restored memory would know.

## Restoring one thing

Usually you do not want a full restore - you want one file back, or one
fact you deleted by mistake.

    # One artifact:
    tar -xzf $BACKUP/artifacts.tar.gz -C /tmp artifacts/<artifact-id>

    # One fact, from the backup database:
    sqlite3 $BACKUP/core.db "SELECT text FROM facts WHERE text LIKE '%sqonion%';"

## Off-site

Local backups do nothing about a dead disk. Set JARVIS_BACKUP_COMMAND
in .env to push each backup somewhere else:

    JARVIS_BACKUP_COMMAND=rclone copy {path} remote:jarvis-backups/
    JARVIS_BACKUP_COMMAND=scp -r {path} jarvis@vps:/opt/jarvis/backups/

Until that is set, copy a backup by hand every few days. Two minutes of
paranoia buys back your assistant's entire memory.
