# Remote source editor

The frontend at `mujocoweb.vercel.app` has a Code editor button. It connects
to `/api/editor` on the currently selected backend. The editor is disabled
unless `EDITOR_TOKEN` is set to at least 32 characters in the local, Git-ignored
`.env`. The current token is stored there, not in the frontend or Git.

The editor allows only these five explicit files:

- Live RoboHive `relocate_v1.py` (environment and rewards)
- Live RoboHive `DAPG_relocate.xml` (default scene)
- Backend copy of `DAPG_relocate.xml` (custom-object template)
- `muj1.py` (web simulation and rendering)
- `mjrlpaul/utils/train_agent.py` (training only; not used by the live demo)

Before the first browser edit, original copies were placed under
`/home/paul/.local/share/mujocoweb-editor-backups`. Each later save also
records the previous version there. The browser can restore the original or
a prior revision. Source writes are atomic, require the current revision hash,
and reject invalid Python/XML syntax. After a change, the local systemd user
service restarts automatically, so the next simulation loads the new code.

The password-protected Backend logs panel shows the last 120 lines from the
local `mujocoweb-backend.service` journal, including Python `print()` output.
It refreshes every three seconds while open. It does not show Render logs or
separate training jobs. `run-local.sh` uses unbuffered Python output so new
prints appear promptly.

**Security warning:** Anyone with the editor token can run arbitrary Python
as the `paul` user on this computer. A syntax check does not detect malicious
code. There is no AI security gate, and one would not make this safe. Keep the
token private, use a strong random token (not `hallo`), and rotate it if it
leaks. The Render deployment should not have `EDITOR_TOKEN` configured.

If the editor itself cannot load, restore a file manually from the backup
directory and restart the service with
`systemctl --user restart mujocoweb-backend`. The backup directory is outside
the Git repository, so a code deploy does not delete it. User services only
run while this user is logged in unless lingering is separately enabled.
